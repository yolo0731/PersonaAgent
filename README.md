# PersonaAgent

PersonaAgent is a Python BotClient plus FastAPI AgentService AI Agent Worker that connects to LiteIM as a normal TCP/TLV client account.

The core boundary is simple:

```text
LiteIM user client
  -> LiteIM C++ server
  -> Python BotClient
  -> FastAPI AgentService
  -> LangGraph workflow
  -> AgentReplyCommand
  -> Python BotClient
  -> LiteIM
  -> user client
```

`BotClient` owns the LiteIM TCP connection. `AgentService` owns the AI workflow. `AgentService` does not hold a LiteIM socket, does not access LiteIM MySQL/Redis, and only returns structured commands for BotClient to execute.

## What It Includes

- Python LiteIM Packet/TLV protocol mirror and frame decoder.
- Async BotClient with login, heartbeat, reconnect, pending request matching, offline message pull, delivery/read ACK, idempotent private replies, and friend policy checks.
- FastAPI AgentService with `/health`, AgentService `/chat` API, and Human Review APIs/UI.
- Six-node LangGraph workflow: `dialogue_policy`, `retrieve_context`, `tool_router`, `generate_reply`, `safety_check`, and `finalize_reply`.
- Knowledge RAG, Memory RAG, and Authorized Style RAG using Chroma plus mock embeddings for tests.
- Consent, PII redaction, style sample import, and verbatim leakage guard for authorized style data.
- Tool Calling framework with schema validation, timeout traces, idempotency keys, and safe memory/context tools.
- Persona prompt engine, structured LLM reply generation, SafetyGuard, Human Review, final `AgentReplyCommand`, and BotClient command execution.
- Evaluation suite with JSONL datasets, offline mock eval, gated real workflow eval, metrics, reports, and failure sample analysis.
- Public runtime examples, tests, and sanitized configuration templates.

## Project Layout

```text
agent_service/          FastAPI service, LangGraph workflow, RAG, memory, style, safety, tools, eval
bot_client/             LiteIM protocol client, runtime, message handling, friend policy, AgentService adapter
scripts/demo/           Offline mock demo
scripts/runtime/        Real BotClient runner
scripts/data/           Authorized style data import/OCR helpers
eval/datasets/          Mock and real-eval JSONL cases
eval/reports/           Generated mock eval reports
tests/                  Pytest coverage for each step and integration boundary
```

## Configuration

Copy the example file and keep real secrets local:

```bash
cp .env.example .env
```

Important runtime variables:

- `LLM_PROVIDER=mock|deepseek`
- `LLM_MODEL=deepseek-v4-flash`
- `OPENAI_API_KEY=...`
- `OPENAI_BASE_URL=https://api.deepseek.com`
- `AGENT_SERVICE_URL=http://127.0.0.1:8088`
- `LITEIM_HOST=127.0.0.1`
- `LITEIM_PORT=9000`
- `BOT_USERNAME=persona_bot`
- `BOT_PASSWORD=...`
- `BOT_STATE_PATH=data/state/bot_state/state.json`
- `AGENT_STATE_DB_PATH=data/state/agent_state/state.sqlite3`
- `MEMORY_DB_PATH=data/state/memory/memory.sqlite3`
- `CHROMA_PATH=data/vector/chroma`
- `STYLE_SAMPLES_PATH=data/authorized_style_records/processed/style_samples.local.jsonl`
- `REAL_EVAL_CONFIRM=0|1`

Real `.env`, runtime databases, local Chroma indexes, raw authorized chat exports, processed local style samples, and runtime logs are ignored by Git.

## Personalized Bot

To configure a personalized chatbot, keep private data in local ignored files and point `.env` at those files:

- Create a normal LiteIM bot account, then set `BOT_USERNAME`, `BOT_PASSWORD`, and `BOT_NICKNAME`.
- Write a local persona config such as `data/authorized_style_records/processed/demo_persona_config.local.yaml`, then set `PERSONA_CONFIG_PATH`.
- Put stable project facts or public knowledge notes under `data/knowledge_docs/` for Knowledge RAG.
- Import only consented and redacted style samples into `STYLE_SAMPLES_PATH`; keep raw chat exports under ignored `data/authorized_style_records/raw/`.
- Set `STYLE_PERSONA_ID`, `STYLE_ON_SMALLTALK=true`, and `STYLE_ON_PRIVATE_CHAT=true` when ordinary private chats should use the configured style.
- Keep `AGENT_STATE_DB_PATH`, `MEMORY_DB_PATH`, `BOT_STATE_PATH`, and `CHROMA_PATH` under ignored runtime/state directories.

The public repository should contain only placeholders, examples, and sanitized source files. Real names, API keys, chat exports, local memories, generated vector indexes, and internal learning notes stay local.

## Quick Start

Install dependencies:

```bash
conda run --no-capture-output -n agent python -m pip install -e ".[dev,openai]"
```

Run the offline mock demo:

```bash
conda run --no-capture-output -n agent python scripts/demo/run_mock_demo.py \
  --output-dir data/runtime/demo
```

Start AgentService in mock mode:

```bash
conda run --no-capture-output -n agent env \
  LLM_PROVIDER=mock \
  LLM_MODEL=mock \
  uvicorn agent_service.main:app --host 127.0.0.1 --port 8088
```

Run the BotClient against LiteIM:

```bash
conda run --no-capture-output -n agent python scripts/runtime/run_bot_client.py \
  --mode agent \
  --username persona_bot \
  --password demo_password \
  --liteim-host 127.0.0.1 \
  --liteim-port 9000 \
  --agent-service-url http://127.0.0.1:8088
```

For a protocol-only smoke path that does not call AgentService:

```bash
conda run --no-capture-output -n agent python scripts/runtime/run_bot_client.py \
  --mode echo \
  --username persona_bot \
  --password demo_password
```

Do not log in to the same bot account from the Qt client while Python BotClient is running.

## Evaluation

Run the default offline eval:

```bash
conda run --no-capture-output -n agent python -m agent_service.eval \
  --mode mock \
  --output-dir eval/reports
```

Real workflow eval is gated to avoid accidental paid model calls:

```bash
conda run --no-capture-output -n agent env \
  REAL_EVAL_CONFIRM=1 \
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  python -m agent_service.eval --mode real --max-cases 20 --concurrency 2 --resume
```

## Tests

```bash
conda run --no-capture-output -n agent python -m pytest -q
conda run --no-capture-output -n agent python -m ruff check .
conda run --no-capture-output -n agent python -m mypy agent_service bot_client
```

## Data Safety

Authorized Style RAG is not ordinary few-shot prompting. Raw chat exports stay local and ignored. Only consent metadata, redacted processed examples, and safe demo artifacts belong in Git. The import pipeline validates consent, applies PII redaction, records import reports, and keeps revocation/active metadata for retrieval filtering.
