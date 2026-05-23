# PersonaAgent

PersonaAgent is project two in the `/home/yolo/jianli` workspace. It is planned as a Python BotClient plus FastAPI AgentService AI Agent Worker.

Step 01 builds only the Python project foundation:

- FastAPI `/health`
- settings loaded from environment variables
- `LLMClient` abstraction
- `MockLLMClient`
- `OpenAILLMClient` shell that is not used by unit tests
- pytest / pytest-asyncio / ruff / mypy configuration

Step 01 does not implement the LiteIM BotClient, LangGraph workflow, RAG, tools, persona, safety, or evaluation.

## Local Test

```bash
conda run -n agent python -m pytest
conda run -n agent ruff check .
conda run -n agent mypy agent_service
```
