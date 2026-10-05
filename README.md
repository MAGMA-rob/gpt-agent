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
