from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
from time import perf_counter
from typing import Any

import httpx
from magma_core.protocol.agent import (
    AgentDecision, AgentError, AgentInput, AgentOutput, AgentRequest, AgentResponse, ToolCall,
)

from .config import Settings
from .prompt import build_system_instruction, build_tools, format_current_input, format_history


RESPONSES_URL = "https://api.openai.com/v1/responses"
GPT_MEMORY_VERSION = 1
CONTEXT_ERROR_TERMS = (
    "context window", "context length", "context limit", "input token limit",
    "token limit", "too many tokens", "exceeds the maximum", "exceeded the maximum",
)


class ProviderFailure(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 rebuildable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.rebuildable = rebuildable


class Runtime:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client
        self.semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def process(self, request: AgentRequest) -> AgentResponse:
        candidates = [
            (entry, index)
            for entry in request.inputs
            for index in range(entry.num_outputs)
        ]
        outputs = await asyncio.gather(*(
            self._process_candidate(request.request_id, entry, index)
            for entry, index in candidates
        ))
        response = AgentResponse(outputs)
        response.validate_request(request)
        return response

    async def _process_candidate(
        self, request_id: str, entry: AgentInput, candidate_index: int
    ) -> AgentOutput:
        memory = deepcopy(entry.memory)
        steps: list[dict[str, Any]] = []
        started_at = perf_counter()
        try:
            history = memory.get("history", [])
            if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
                raise ValueError("memory.history must be a list of objects")
            state = memory.get("_gpt", {})
            if not isinstance(state, dict):
                raise ValueError("memory._gpt must be an object")
            if state and state.get("version") != GPT_MEMORY_VERSION:
                raise ValueError("Unsupported memory._gpt version")
            previous_id = state.get("response_id")
            if previous_id is not None and (not isinstance(previous_id, str) or not previous_id):
                raise ValueError("memory._gpt.response_id must be a non-empty string")
            pending_calls = state.get("pending_calls", [])
            if not isinstance(pending_calls, list) or any(
                not isinstance(call, dict)
                or not isinstance(call.get("name"), str)
                or not call.get("name")
                or not isinstance(call.get("call_id"), str)
                or not call.get("call_id")
                for call in pending_calls
            ):
                raise ValueError("memory._gpt.pending_calls must contain name/call_id objects")

            system_instruction = build_system_instruction(memory)
            tools = build_tools(entry.tools, entry.attributes)
            current_input = format_current_input(
                entry.attributes, entry.instruction.type, entry.instruction.content
            )
            reconstructed = not previous_id or (bool(pending_calls) and entry.instruction.type != "env")
            if reconstructed:
                transcript = format_history(history)
                input_value: str | list[dict[str, Any]] = (
                    f"{transcript}\n\n{current_input}" if transcript else current_input
                )
                previous_id = None
            elif pending_calls:
                input_value = [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": current_input,
                    }
                    for call in pending_calls
                ]
            else:
                input_value = current_input

            payload: dict[str, Any] = {
                "model": self.settings.model,
                "store": True,
                "instructions": system_instruction,
                "reasoning": {
                    "effort": self.settings.reasoning_effort,
                    "context": "all_turns",
                },
                "input": input_value,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
                payload["parallel_tool_calls"] = True
            if previous_id:
                payload["previous_response_id"] = previous_id

            trace: dict[str, Any] = {
                "id": "gpt-0",
                "component": "gpt",
                "origin": "model",
                "full_prompt": json.dumps(payload, ensure_ascii=False, indent=2),
                "input_elements": {
                    "context_source": (
                        "previous_response_id" if previous_id else "memory.history"
                    ),
                    "history": deepcopy(history),
                    "request": deepcopy(payload),
                },
                "output_raw": "",
            }
            if state.get("response_id") and reconstructed:
                trace["input_elements"]["reconstruction_reason"] = "user_input_with_pending_calls"
            steps.append(trace)
            try:
                provider_response = await self._send_with_retries(payload, trace)
            except ProviderFailure as error:
                if not error.rebuildable or not previous_id:
                    raise
                trace["error"] = str(error)
                transcript = format_history(history)
                payload.pop("previous_response_id", None)
                payload["input"] = (
                    f"{transcript}\n\n{current_input}" if transcript else current_input
                )
                trace = {
                    "id": "gpt-1",
                    "component": "gpt",
                    "origin": "model",
                    "full_prompt": json.dumps(payload, ensure_ascii=False, indent=2),
                    "input_elements": {
                        "context_source": "memory.history",
                        "history": deepcopy(history),
                        "request": deepcopy(payload),
                        "reconstruction_reason": error.code,
                    },
                    "output_raw": "",
                }
                steps.append(trace)
                provider_response = await self._send_with_retries(payload, trace)

            if not isinstance(provider_response, dict):
                raise ValueError("OpenAI response must be an object")
            response_id = provider_response.get("id")
            if not isinstance(response_id, str) or not response_id:
                raise ValueError("OpenAI response has no response id")

            usage = provider_response.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError("OpenAI usage must be an object")
            input_details = usage.get("input_tokens_details") or {}
            output_details = usage.get("output_tokens_details") or {}
            if not isinstance(input_details, dict) or not isinstance(output_details, dict):
                raise ValueError("OpenAI token details must be objects")
            trace.update({
                "response_id": response_id,
                "total_input_tokens": usage.get("input_tokens"),
                "total_cached_tokens": input_details.get("cached_tokens"),
                "total_output_tokens": usage.get("output_tokens"),
                "total_thought_tokens": output_details.get("reasoning_tokens"),
                "usage": usage,
            })
            raw_steps = provider_response.get("output")
            if not isinstance(raw_steps, list):
                raise ValueError("OpenAI output must be a list")
            visible_parts: list[str] = []
            calls: list[ToolCall] = []
            next_pending: list[dict[str, str]] = []
            for step in raw_steps:
                if not isinstance(step, dict):
                    raise ValueError("OpenAI step must be an object")
                if step.get("type") == "message":
                    if step.get("status") not in {None, "completed"}:
                        raise ValueError("OpenAI message is not completed")
                    content = step.get("content")
                    if not isinstance(content, list):
                        raise ValueError("OpenAI message content must be a list")
                    for part in content:
                        if not isinstance(part, dict):
                            raise ValueError("OpenAI message content must contain objects")
                        if part.get("type") == "refusal":
                            raise ProviderFailure("model_refusal", str(part.get("refusal") or "OpenAI refused the request"))
                        if part.get("type") == "output_text":
                            value = part.get("text")
                            if not isinstance(value, str):
                                raise ValueError("OpenAI output text must be a string")
                            visible_parts.append(value)
                elif step.get("type") == "function_call":
                    if step.get("status") not in {None, "completed"}:
                        raise ValueError("OpenAI function call is not completed")
                    name = step.get("name")
                    raw_arguments = step.get("arguments")
                    call_id = step.get("call_id")
                    if not isinstance(name, str) or not name:
                        raise ValueError("OpenAI function call has no name")
                    if not isinstance(raw_arguments, str):
                        raise ValueError(f"OpenAI function call {name} arguments must be JSON text")
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError(f"OpenAI function call {name} has invalid arguments")
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError(f"OpenAI function call {name} has no call_id")
                    target_robot = arguments.get("target_robot")
                    if not isinstance(target_robot, str) or not target_robot:
                        raise ValueError(f"OpenAI function call {name} has no target_robot")
                    tool_arguments = dict(arguments)
                    tool_arguments.pop("target_robot")
                    calls.append(ToolCall(
                        name=name, arguments=tool_arguments,
                        target_robot_name=target_robot,
                    ))
                    next_pending.append({"name": name, "call_id": call_id})
                elif step.get("type") != "reasoning":
                    raise ValueError(f"Unsupported OpenAI output type: {step.get('type')}")

            say = "".join(visible_parts)
            if not say.strip() and not calls:
                raise ValueError("OpenAI response contains no text or function call")
            decision = AgentDecision(say=say, tool_calls=calls)
            trace["output_formatted"] = decision.model_dump(mode="json")
            memory["_gpt"] = {
                "version": GPT_MEMORY_VERSION,
                "response_id": response_id,
                "pending_calls": next_pending,
            }
            action = [
                {call.target_robot_name: {"name": call.name, "arguments": call.arguments}}
                for call in calls
            ]
            memory.setdefault("history", []).extend([
                {
                    "author": "USER" if entry.instruction.type == "user" else "SYSTEM",
                    "content": entry.instruction.content,
                },
                {
                    "author": "MODEL",
                    "content": json.dumps({"say": say, "action": action}, ensure_ascii=False),
                },
            ])
            trace["candidate_duration_seconds"] = perf_counter() - started_at
            return AgentOutput(
                request_id=request_id,
                source_id=entry.id,
                candidate_index=candidate_index,
                status="completed",
                memory=memory,
                internal_steps=steps,
                output=decision,
            )
        except Exception as error:
            if isinstance(error, ProviderFailure):
                code = error.code
            elif isinstance(error, (TypeError, ValueError, KeyError)):
                code = "invalid_output"
            else:
                code = "internal_error"
                logging.exception("OpenAI candidate processing failed")
            if steps:
                steps[-1]["error"] = str(error)
                steps[-1]["candidate_duration_seconds"] = perf_counter() - started_at
            else:
                steps.append({"component": "gpt", "origin": "model", "error": str(error)})
            return AgentOutput(
                request_id=request_id,
                source_id=entry.id,
                candidate_index=candidate_index,
                status="error",
                memory=deepcopy(entry.memory),
                internal_steps=steps,
                error=AgentError(code=code, message=str(error) or code, component="gpt"),
            )

    async def _send_with_retries(
        self, payload: dict[str, Any], trace: dict[str, Any]
    ) -> dict[str, Any]:
        started_at = perf_counter()
        try:
            for attempt in range(self.settings.max_retries + 1):
                attempt_trace: dict[str, Any] = {"attempt": attempt + 1}
                trace.setdefault("attempts", []).append(attempt_trace)
                attempt_started_at = perf_counter()
                try:
                    async with self.semaphore:
                        response = await self.client.post(
                            RESPONSES_URL,
                            json=payload,
                            timeout=self.settings.request_timeout_seconds,
                        )
                except httpx.RequestError as error:
                    failure = ProviderFailure(
                        "transport_error", str(error) or type(error).__name__,
                        retryable=True, rebuildable=True,
                    )
                else:
                    trace["http_status"] = response.status_code
                    trace["output_raw"] = response.text
                    attempt_trace.update({
                        "http_status": response.status_code,
                        "output_raw": response.text,
                    })
                    if response.is_success:
                        try:
                            provider_response = response.json()
                        except ValueError as error:
                            raise ProviderFailure("invalid_response", "OpenAI returned invalid JSON") from error
                        if not isinstance(provider_response, dict):
                            raise ProviderFailure("invalid_response", "OpenAI returned a non-object response")
                        status = provider_response.get("status")
                        if status != "completed":
                            details = provider_response.get("error") or provider_response.get("incomplete_details")
                            message = json.dumps(details, ensure_ascii=False) if details else f"OpenAI status: {status}"
                            context_exceeded = any(
                                term in message.lower() for term in CONTEXT_ERROR_TERMS
                            ) or "context_length_exceeded" in message.lower()
                            raise ProviderFailure(
                                "context_exceeded" if context_exceeded else "provider_status",
                                message, rebuildable=context_exceeded,
                            )
                        return provider_response

                    try:
                        body = response.json()
                    except ValueError:
                        body = {}
                    provider_error = body.get("error", {}) if isinstance(body, dict) else {}
                    if not isinstance(provider_error, dict):
                        provider_error = {}
                    attempt_trace["provider_error"] = provider_error
                    message = str(provider_error.get("message") or f"OpenAI HTTP {response.status_code}")
                    provider_code = str(provider_error.get("code") or "")
                    parameter = str(provider_error.get("param") or "")
                    lowered = message.lower()
                    # Only context errors can repair a 4xx; auth/schema/model errors stay fatal.
                    context_exceeded = response.status_code == 400 and (
                        provider_code == "context_length_exceeded"
                        or any(term in lowered for term in CONTEXT_ERROR_TERMS)
                    )
                    invalid_id = bool(payload.get("previous_response_id")) and response.status_code in {400, 404} and (
                        provider_code == "previous_response_not_found"
                        or parameter == "previous_response_id"
                        or (any(term in lowered for term in (
                            "previous response", "previous_response_id", "response with id", "response 'resp_", "response \"resp_",
                        )) and any(
                            term in lowered for term in ("not found", "invalid", "expired", "does not exist")
                        ))
                    )
                    invalid_calls = bool(payload.get("previous_response_id")) and response.status_code == 400 and (
                        "no tool output found" in lowered
                        or "no tool call found" in lowered
                        or (any(term in lowered for term in ("call_id", "function call", "function_call")) and any(
                            term in lowered for term in ("not found", "missing", "no tool", "no output", "no function", "mismatch")
                        ))
                    )
                    retryable = response.status_code == 429 or response.status_code >= 500
                    failure = ProviderFailure(
                        "context_exceeded" if context_exceeded else
                        "invalid_response_id" if invalid_id else
                        "invalid_tool_context" if invalid_calls else
                        "provider_error",
                        message, retryable=retryable,
                        rebuildable=context_exceeded or invalid_id or invalid_calls or retryable,
                    )
                finally:
                    attempt_trace["duration_seconds"] = perf_counter() - attempt_started_at

                attempt_trace["error"] = str(failure)
                if not failure.retryable or attempt == self.settings.max_retries:
                    raise failure
                delay = min(2 ** attempt, 8)
                trace.setdefault("retries", []).append({
                    "attempt": attempt + 1,
                    "error": str(failure),
                    "delay_seconds": delay,
                })
                await asyncio.sleep(delay)
            raise RuntimeError("Unreachable retry state")
        finally:
            trace["duration_seconds"] = perf_counter() - started_at
