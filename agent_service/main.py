from fastapi import FastAPI

from agent_service.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings()
    app = FastAPI(title="PersonaAgent AgentService")
    app.state.settings = app_settings

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": app_settings.service_name}

    return app


app = create_app()
