from fastapi import FastAPI  # 导入的是 FastAPI 的核心类

from agent_service.config import Settings  # 导入我们定义的 Settings 类，用于读取和管理运行时配置

"""这个文件负责创建 AgentService 的 Web 应用，并提供一个 /health 健康检查接口。"""

# settings 可以是 Settings 也可以是 None
def create_app(settings: Settings | None = None) -> FastAPI:
    # 如果外部没有传入 settings，就创建一个默认的 Settings 实例
    app_settings = settings or Settings()
    # 创建一个 FastAPI 应用，并将 settings 存储在 app.state 中
    app = FastAPI(title="PersonaAgent AgentService")
    app.state.settings = app_settings

    # 装饰器定义了一个 GET 请求的 /health 路径，访问这个路径会调用 health 函数
    @app.get("/health")
    def health() -> dict[str, str]:  # 嵌套函数，返回一个字典（键值对）
        return {"status": "ok", "service": app_settings.service_name}

    # 返回配置好的 app
    return app


app = create_app()  # 创建一个全局的 FastAPI 应用实例
