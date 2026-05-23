"""FastAPI entrypoint for AgentService."""

from fastapi import FastAPI

from agent_service.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings()
    app = FastAPI(title="PersonaAgent AgentService")

    # Settings 挂在 app.state 上，后续 /chat、LLM、trace 都从这里读取运行配置。
    app.state.settings = app_settings

    @app.get("/health")
    def health() -> dict[str, str]:
        # /health 只证明服务进程可用，不触发 DeepSeek、RAG 或 LiteIM 依赖。
        return {"status": "ok", "service": app_settings.service_name}

    return app


app = create_app()
