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
- pytest / pytest-asyncio / ruff / mypy configuration

The project still does not implement live LiteIM network connection, BotClient login/reconnect, LangGraph workflow, RAG, tools, persona, safety, or evaluation.

## Local Runtime Config

PersonaAgent uses DeepSeek as the default real LLM provider through the OpenAI SDK compatible API.

Create local `.env` from `.env.example`, then put your real DeepSeek key in `OPENAI_API_KEY`. Do not commit `.env`.

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
OPENAI_API_KEY=replace_with_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com
```

Unit tests still use `MockLLMClient` and do not call DeepSeek.

## Local Test

```bash
conda run -n agent python -m pytest
conda run -n agent ruff check .
conda run -n agent mypy agent_service bot_client
```
