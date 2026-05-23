# PersonaAgent

PersonaAgent is project two in the `/home/yolo/jianli` workspace. It is planned as a Python BotClient plus FastAPI AgentService AI Agent Worker.

Current implemented foundation:

- FastAPI `/health`
- settings loaded from environment variables
- `LLMClient` abstraction
- `MockLLMClient`
- `OpenAILLMClient` shell that is not used by unit tests
- Python LiteIM V1 Packet/TLV protocol mirror
- Python LiteIM V1 `FrameDecoder` for half-packet, sticky-packet, and error-state handling
- pytest / pytest-asyncio / ruff / mypy configuration

The project still does not implement live LiteIM network connection, BotClient login/reconnect, LangGraph workflow, RAG, tools, persona, safety, or evaluation.

## Local Test

```bash
conda run -n agent python -m pytest
conda run -n agent ruff check .
conda run -n agent mypy agent_service bot_client
```
