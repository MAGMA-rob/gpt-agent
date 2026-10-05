from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import httpx
from magma_core.protocol.agent import AgentHealth, AgentInfo, AgentRequest, AgentResponse
import uvicorn

from . import __version__
from .config import Settings
from .runtime import Runtime


def create_app(settings: Settings, client: httpx.AsyncClient | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        api_key = os.environ.get("OPENAI_API_KEY")
        if client is None:
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required")
            active_client = httpx.AsyncClient(headers={"Authorization": f"Bearer {api_key}"})
        else:
            active_client = client
        app.state.runtime = Runtime(settings, active_client)
        try:
            yield
        finally:
            app.state.runtime = None
            if client is None:
                await active_client.aclose()

    app = FastAPI(title="gpt-agent", version=__version__, lifespan=lifespan)

    @app.get("/health", response_model=AgentHealth)
    async def health() -> AgentHealth:
        if getattr(app.state, "runtime", None) is None:
            raise HTTPException(status_code=503, detail="Agent is not ready")
        return AgentHealth()

    @app.get("/v1/info", response_model=AgentInfo)
    async def info() -> AgentInfo:
        return AgentInfo(
            agent_id="gpt-agent",
            agent_version=__version__,
            capabilities={"inference": True, "coaching": False},
            coaching_unavailable_reason="Coaching is not available in gpt-agent",
        )

    @app.post("/v1/responses", response_model=AgentResponse)
    async def responses(request: AgentRequest) -> AgentResponse:
        runtime: Runtime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="Agent is not ready")
        return await runtime.process(request)

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gpt-agent")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--request-timeout-seconds", type=float)
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")
        for field in (
            "host", "port", "model", "reasoning_effort", "max_concurrency",
            "max_retries", "request_timeout_seconds",
        ):
            value = getattr(args, field)
            if value is not None:
                config[field] = value
        settings = Settings.model_validate(config)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
