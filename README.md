# PersonaAgent

PersonaAgent is project two in the `/home/yolo/jianli` workspace. It is planned as a Python BotClient plus FastAPI AgentService AI Agent Worker.

Current implemented foundation:

- FastAPI `/health`
- settings loaded from environment variables
- `LLMClient` abstraction
- `MockLLMClient`
- OpenAI SDK compatible LLM client, configured for DeepSeek in real runtime mode
- Python LiteIM V1 Packet/TLV protocol mirror
- Python LiteIM V1 `FrameDecoder` for half-packet, sticky-packet, and error-state handling
- Cross-language LiteIM protocol contract tests against the sibling C++ protocol implementation
- Async LiteIM `BotClient` with TCP connect, login/register helpers, pending request matching, timeout cleanup, heartbeat, close/logout, and supervisor reconnect
- LiteIM reliability helpers for offline pull/ACK, delivery ACK, read ACK, `ClientMessageId` replies, local message deduplication, and receipt trace storage
- Friend request policy helpers for allowlisted Agent access, automatic accept/reject decisions, friend-list sync, accepted-friend pushes, and non-friend private-message blocking
- Reliable Echo mode runtime that connects/login, syncs friends, processes offline messages, consumes live pushes, replies to private chats through `ClientMessageId`, and records group pushes without replying
- AgentService `/chat` API with `ChatRequest`, `AgentReplyCommand`, structured error envelopes, and a mock reply handler
- BotClient-side AgentService adapter that converts LiteIM messages into `/chat` requests and fails closed on AgentService timeout or malformed responses
- LangGraph six-node Agent workflow skeleton with deterministic mock nodes, no-reply early finalize, safety-block no-send behavior, and per-node trace state
- DialoguePolicy structured decision schema with mock structured-output validation, retry, fallback rules, intent classification, and need flags for knowledge, memory, style, tools, and human review
- SQLite-backed checkpoint and Human Review skeleton with pending review storage, approve/reject/edit/resume APIs, thread IDs, and idempotent resume no-send behavior
- Knowledge RAG pipeline with document loading, recursive chunking, mock embeddings, persistent Chroma collection, active metadata filtering, top-k retrieval, and workflow trace integration
- Memory RAG pipeline with SQLite memory records, Chroma memory collection, user-scoped retrieval, active filtering, `/remember`, `/forget`, and workflow context injection
- Authorized data governance pipeline with consent manifest validation, PII redaction, processed style sample import, revocation-aware active status, ignored raw data folders, and import reports
- Authorized Style RAG pipeline with a dedicated Chroma style collection, `persona_id` / `consent_id` / `active` filtering, deterministic style feature extraction, insufficient-sample fallback, and workflow trace/context injection
- Verbatim leakage guard with n-gram overlap, longest common substring ratio, PII second scan, style source-id checks, deterministic rewrite/block actions, workflow safety integration, and leakage metrics
- Tool Calling framework with a registry, Pydantic input/output schemas, timeout trace, structured error envelopes, idempotency keys for side-effect tools, safe memory/profile/context tools, and workflow state integration
- PersonaEngine with versioned `persona.yaml`, required identity notice, prompt templates, style instruction, safety boundaries, prompt metadata, and used context ids
- pytest / pytest-asyncio / ruff / mypy configuration

The project still does not implement real LLM structured output, real reply generation, broader production safety policy, a human review UI, or evaluation.

## Local Runtime Config

PersonaAgent uses DeepSeek as the default real LLM provider through the OpenAI SDK compatible API.

Create local `.env` from `.env.example`, then put your real DeepSeek key in `OPENAI_API_KEY`. Do not commit `.env`.

```env
AGENT_HOST=127.0.0.1
AGENT_PORT=8088
AGENT_SERVICE_URL=http://127.0.0.1:8088
AGENT_REQUEST_TIMEOUT_SECONDS=5.0
AGENT_STATE_DB_PATH=data/agent_state/state.sqlite3
CHROMA_PATH=data/chroma
MEMORY_DB_PATH=data/memory/memory.sqlite3
MEMORY_TOP_K=5
KNOWLEDGE_DOCS_PATH=data/knowledge_docs
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=5
STYLE_TOP_K=8
PERSONA_CONFIG_PATH=agent_service/persona/persona.yaml
LITEIM_HOST=127.0.0.1
LITEIM_PORT=9000
BOT_USERNAME=persona_agent_bot
BOT_PASSWORD=change_me
BOT_NICKNAME=PersonaAgent
BOT_STATE_PATH=data/bot_state/state.json
BOT_OFFLINE_MESSAGE_LIMIT=100
BOT_ALLOWED_USER_IDS=
BOT_ALLOWED_USERNAMES=
BOT_AUTO_ACCEPT_FRIEND_REQUESTS=true
BOT_REJECT_NON_ALLOWLISTED_FRIEND_REQUESTS=true
ECHO_MODE=true
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
OPENAI_API_KEY=replace_with_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com
```

`BotClient` connects to LiteIM as a normal user over the same TCP/TLV protocol. `AgentService` does not hold the LiteIM TCP connection and does not directly send LiteIM packets.

`BOT_ALLOWED_USER_IDS` and `BOT_ALLOWED_USERNAMES` restrict who can become friends with the Agent account. By default, allowlisted requests are accepted and non-allowlisted requests are rejected.

`ECHO_MODE=true` enables the Step 07 smoke path: BotClient replies to private messages with the same text after delivery/read ACK and message deduplication. It does not call DeepSeek or AgentService.

`AGENT_SERVICE_URL` and `AGENT_REQUEST_TIMEOUT_SECONDS` configure the Step 08 BotClient adapter. If AgentService is unavailable, times out, returns an HTTP error, or returns a malformed response, the adapter returns `should_send=false` and BotClient does not send a LiteIM message.

`AGENT_STATE_DB_PATH` stores AgentService checkpoint and Human Review state. Keep the real SQLite database ignored; only `data/agent_state/.gitignore` is tracked.

`CHROMA_PATH`, `KNOWLEDGE_DOCS_PATH`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, and `RAG_TOP_K` configure the Step 12 Knowledge RAG pipeline. Keep local Chroma data ignored; only `data/chroma/.gitignore` is tracked. Default tests use `MockEmbeddingClient` and do not call a real embedding API.

`MEMORY_DB_PATH` and `MEMORY_TOP_K` configure the Step 13 Memory RAG pipeline. Memory records are scoped by `sender_id` as `user_id`; local memory SQLite data stays ignored, and only `data/memory/.gitignore` is tracked.

`data/authorized_style_records/raw/` is reserved for local authorized chat exports and is ignored by Git. Step 14 tracks only safe examples: consent manifest, redacted processed style samples, and import reports under `data/authorized_style_records/`.

`STYLE_TOP_K` configures the Step 15 Authorized Style RAG retrieval limit. Style retrieval uses the `style` collection under `CHROMA_PATH`, derives the target `persona_id` from `sender_id` for current-user style requests, and only injects processed, redacted style examples into workflow context.

Step 16 adds a deterministic Verbatim Leakage Guard after draft generation and before finalization. Direct style-sample copies and PII leaks are blocked, high-overlap drafts are rewritten to a safe fallback, and normal style-similar but non-verbatim replies can pass.

Step 17 adds a safe Tool Calling framework inside AgentService. Built-in tools include `save_memory`, `deactivate_memory`, `get_user_profile`, `summarize_recent_context`, `search_recent_context`, and `liteim_context_tool`. Side-effect tools require an `idempotency_key`, tool failures are returned as structured envelopes, and tools do not send LiteIM messages or access LiteIM MySQL/TCP.

`PERSONA_CONFIG_PATH` points to the Step 18 persona and prompt template config. The default `agent_service/persona/persona.yaml` keeps identity notice, prompt version, style instruction, and safety boundaries out of Python code. `PersonaEngine` injects that identity notice into prompts and records prompt metadata plus `used_context_ids` in workflow state.

`BOT_STATE_PATH` stores local processed-message IDs, delivery/read receipt traces, synced friends, friend policy traces, and group-message trace records. Keep the real runtime state ignored; only `data/bot_state/.gitignore` is tracked.

Unit tests still use `MockLLMClient` and do not call DeepSeek.

## Local Test

```bash
conda run -n agent python -m pytest
conda run -n agent ruff check .
conda run -n agent mypy agent_service bot_client
```

The cross-language protocol tests compile a small C++ helper into pytest's temporary directory and link it against `/home/yolo/jianli/LiteIM` protocol sources. They do not start the LiteIM server.

The Step 04 BotClient tests use an in-process asyncio mock LiteIM server. They do not require MySQL, Redis, or a running LiteIM server.

The Step 05 reliability tests use protocol packets and fake BotClient objects to verify ACK/order/dedup behavior without calling AgentService.

The Step 06 friend policy tests use protocol packets and fake BotClient objects to verify allowlist accept/reject behavior, friend-list sync, accepted-push handling, and non-friend private-message blocking.

The Step 07 Echo runtime tests verify startup sync order, offline echo-once behavior, live private echo with ACK/read/reply, restart deduplication, echo disabled behavior, and group push record-without-reply behavior.

The Step 08 chat API adapter tests verify `/chat` mock replies, structured error envelopes, LiteIM message to `ChatRequest` mapping, AgentApiClient success/fail-closed behavior, and `should_send=false` no-send behavior.

The Step 09 LangGraph workflow tests verify full six-node graph execution, no-reply early finalize, safety-block no-send behavior, node trace recording, and `/chat` default graph integration.

The Step 10 DialoguePolicy tests verify the structured decision schema, all supported intents, private-chat default reply, group-chat no-op, mock structured-output retry, fallback rules, workflow routing, and `/chat` group no-op behavior.

The Step 11 Human Review tests verify thread ID construction, high-risk pending review and checkpoint persistence, approve/edit/resume, reject/resume no-op, and repeated resume no-send behavior.

The Step 12 Knowledge RAG tests verify document loading, required metadata, Chroma persistence, top-k retrieval, active metadata filtering, empty collection behavior, and workflow context/trace integration.

The Step 13 Memory RAG tests verify memory save/list/deactivate fields, user-scoped retrieval, inactive filtering, `/remember`, `/forget`, and memory query context injection.

The Step 14 governance tests verify no-consent rejection, forbidden usage rejection, PII redaction before processed sample writing, revoked-consent inactive samples, import report generation, and the safe authorized-style data layout.

The Step 15 Authorized Style RAG tests verify `persona_id` isolation, `consent_id` filtering, `active=false` exclusion, deterministic style feature stats, insufficient-sample fallback, and workflow style trace/context injection.

The Step 16 Verbatim Leakage Guard tests verify direct style-sample copy blocking, high-overlap rewrite, PII leak blocking, style source-id leak blocking, normal non-verbatim pass behavior, and workflow safety blocking from retrieved style context.

The Step 17 Tool Calling tests verify registry schema validation, timeout trace recording, required built-in tool registration, `save_memory` idempotency, structured tool failure envelopes, and workflow tool-result state injection.

The Step 18 PersonaEngine tests verify default `persona.yaml` loading, required identity notice validation, context-aware prompt assembly, style fallback behavior, prompt metadata, and prompt version workflow trace recording.
