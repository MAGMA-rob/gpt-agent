# GPT Agent

`gpt-agent` is an inference-only MAGMA protocol-v2 agent using `gpt-6.1-sol`
through OpenAI's Responses API. Its server and execution loop follow
`gemini-agent`; it does not implement coaching or export.

## Install and run

Use Python 3.12 in an environment with MAGMA Core:

```bash
python -m pip install -e agents/gpt-agent
export OPENAI_API_KEY='your-key'
gpt-agent --host 127.0.0.1 --port 8888 \
  --reasoning-effort medium --max-concurrency 4
```

`python -m gpt_agent` provides the same CLI. The key is read only from
`OPENAI_API_KEY` and sent in the Authorization header, never in the trace.
Point MAGMA at `http://127.0.0.1:8888`. `/health`, `/v1/info` and
`/v1/responses` use MAGMA protocol version 2.0.

Defaults are model `gpt-6.1-sol`, medium reasoning, four concurrent API requests,
two retries for temporary failures, and a 90-second HTTP timeout per attempt.
Configure them with `--model`, `--reasoning-effort`, `--max-concurrency`,
`--max-retries`, `--request-timeout-seconds`, `--host`, and `--port`.
`--config` accepts a JSON object with the corresponding snake_case field names;
CLI arguments override the file. Reasoning efforts are `low`, `medium`, `high`,
`xhigh`, and `max`.

The agent sends no temperature, top-p, seed, or greedy-decoding parameters.
The benchmark's deterministic-decoding option does not make this provider
deterministic. See [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
and [GPT-6 API compatibility](https://developers.openai.com/api/docs/guides/latest-model#migration-quickstart).

## Tools and conversation state

All declared parameters are required, including `target_robot`. Complete
schemas use strict function calling. Tools containing open dictionaries or
arrays with unspecified element schemas explicitly use `strict: false`, while
keeping every parameter required. Descriptions and nested constraints are
preserved; the agent does not invent internal schemas for these structures.

The model is instructed to call tools without a textual preamble and to use
parallel calls only for independent actions on different robots. The agent
returns native calls as MAGMA `tool_calls`; it never executes robot tools and
does not check whether multiple calls target the same robot. The engine handles
execution and action validation.

Each candidate returns its own `memory._gpt` containing a version, the last
`response_id`, and pending function names and `call_id` values. MAGMA must pass
that memory back with the next input. Candidates have independent memory even
when they branch from the same previous response.

Requests use `store: true`, `previous_response_id`, and
`reasoning.context: "all_turns"`. OpenAI retains the earlier context and
reasoning; the agent does not keep or clear reasoning locally. Instructions,
permanent rules, and current tools are sent on every decision. Stored context
remains subject to provider retention and model context limits, and chained
input tokens still incur API costs. See [conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
and [persisted reasoning](https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-across-calls).

For an environment feedback input, the same global MAGMA feedback and current
attributes are returned to every pending `call_id` as `function_call_output`.
No individual execution result is invented. If a new user instruction arrives
while tool calls remain pending, the agent starts a conversation reconstructed
from the readable history, as Gemini does.

## Retries, reconstruction and traces

Network errors, timeouts, HTTP 429 and HTTP 5xx receive up to two retries by
default, with delays of one and two seconds. Each retry resends the same payload.
HTTP requests share a concurrency semaphore; retry delays do not hold it.
Authentication, permission, model and schema errors are returned directly.

On an active response chain, the agent tries one reconstruction if the response
ID is unavailable, context is too long, tool-result linkage is rejected, or
temporary retries are exhausted. It removes `previous_response_id` and
concatenates all of `memory.history` with the current input, without reasoning
or old native tool-result items. That request has its own retry budget. On
success, subsequent decisions continue from its new response ID.

Reconstruction does not remove history entries. If the rebuilt input is still
too large, or reconstruction otherwise fails, the candidate returns a protocol
error with its original memory. A request already built without a previous ID
does not attempt an identical reconstruction. Incomplete, refused or malformed
outputs are errors and never return partial actions for execution.

`memory.history` is a readable fallback, updated exactly once per successful
decision. `internal_steps` records request bodies, each HTTP attempt and response
body, retry delays, reconstruction reasons, converted decisions, response IDs,
durations and token usage (input, cached input, output and reasoning).
Reconstruction gets a separate trace step. Continuation requests contain the
previous response ID, not the full context stored by OpenAI; readable history is
included alongside the request for inspection.

## Benchmark

After starting the agent with your API key, run your compiled benchmark:

```bash
magma-bench run \
  --benchmark-root /path/to/compiled-benchmark \
  --results-path ./eval/gpt-6.1-sol-medium \
  --agent-address http://127.0.0.1:8888 \
  --agent-timeout 600 \
  --nb-env 4 \
  --model-logs \
  --no-deterministic-decoding \
  --run-name gpt-6.1-sol-medium
```

With default settings, 600 seconds covers two sequences of three HTTP attempts
at 90 seconds each, plus retry delays. Semaphore queueing can require additional
time. Increase the benchmark timeout if you increase concurrency load, attempt
timeouts, or retry counts. Failed candidates remain independent of successful
candidates.

Implementation validation uses simulated HTTP responses without an API key.
Live model access and benchmark performance must be evaluated separately with
your credentials.
