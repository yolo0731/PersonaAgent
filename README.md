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
- pytest / pytest-asyncio / ruff / mypy configuration

The project still does not implement LangGraph workflow, RAG, tools, persona, safety, real reply generation, or evaluation.

## Local Runtime Config

PersonaAgent uses DeepSeek as the default real LLM provider through the OpenAI SDK compatible API.

Create local `.env` from `.env.example`, then put your real DeepSeek key in `OPENAI_API_KEY`. Do not commit `.env`.

```env
AGENT_HOST=127.0.0.1
AGENT_PORT=8088
AGENT_SERVICE_URL=http://127.0.0.1:8088
AGENT_REQUEST_TIMEOUT_SECONDS=5.0
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
