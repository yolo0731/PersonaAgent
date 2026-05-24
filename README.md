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
- pytest / pytest-asyncio / ruff / mypy configuration

The project still does not implement business-message handling, AgentService-to-BotClient command dispatch, LangGraph workflow, RAG, tools, persona, safety, or evaluation.

## Local Runtime Config

PersonaAgent uses DeepSeek as the default real LLM provider through the OpenAI SDK compatible API.

Create local `.env` from `.env.example`, then put your real DeepSeek key in `OPENAI_API_KEY`. Do not commit `.env`.

```env
LITEIM_HOST=127.0.0.1
LITEIM_PORT=9000
BOT_USERNAME=persona_agent_bot
BOT_PASSWORD=change_me
BOT_NICKNAME=PersonaAgent
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
OPENAI_API_KEY=replace_with_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com
```

`BotClient` connects to LiteIM as a normal user over the same TCP/TLV protocol. `AgentService` does not hold the LiteIM TCP connection and does not directly send LiteIM packets.

Unit tests still use `MockLLMClient` and do not call DeepSeek.

## Local Test

```bash
conda run -n agent python -m pytest
conda run -n agent ruff check .
conda run -n agent mypy agent_service bot_client
```

The cross-language protocol tests compile a small C++ helper into pytest's temporary directory and link it against `/home/yolo/jianli/LiteIM` protocol sources. They do not start the LiteIM server.

The Step 04 BotClient tests use an in-process asyncio mock LiteIM server. They do not require MySQL, Redis, or a running LiteIM server.
