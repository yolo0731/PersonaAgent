from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings 按优先级查找每个字段的值：
# 环境变量（如 AGENT_HOST）> .env 文件 > Field(default=...) 的默认值


class Settings(BaseSettings):
    """Runtime settings for the PersonaAgent service."""

    # 服务监听地址
    service_name: str = "personaagent"
    agent_host: str = Field(default="127.0.0.1", validation_alias="AGENT_HOST")
    agent_port: int = Field(default=8088, validation_alias="AGENT_PORT")

    # LLM 提供商和模型
    llm_provider: str = Field(default="mock", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="mock", validation_alias="LLM_MODEL")

    # DeepSeek 通过 OpenAI SDK 兼容协议接入，因此沿用 OPENAI_* 环境变量名。
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")

    # Embedding 模型和提供商
    embedding_provider: str = Field(default="openai", validation_alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
    )
    # 向量数据库类型
    vector_db: str = Field(default="chroma", validation_alias="VECTOR_DB")
    chroma_path: str = Field(default="data/chroma", validation_alias="CHROMA_PATH")
    memory_db_path: str = Field(
        default="data/memory/memory.sqlite3",
        validation_alias="MEMORY_DB_PATH",
    )
    memory_top_k: int = Field(default=5, validation_alias="MEMORY_TOP_K")
    knowledge_docs_path: str = Field(
        default="data/knowledge_docs",
        validation_alias="KNOWLEDGE_DOCS_PATH",
    )
    rag_chunk_size: int = Field(default=500, validation_alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=50, validation_alias="RAG_CHUNK_OVERLAP")
    # RAG 和风格检索配置
    rag_top_k: int = Field(default=5, validation_alias="RAG_TOP_K")
    # 风格样本检索数量
    style_top_k: int = Field(default=8, validation_alias="STYLE_TOP_K")

    # 是否开启 echo 模式（返回 LLM 原始输出）
    echo_mode: bool = Field(default=True, validation_alias="ECHO_MODE")
    # 是否开启 LLM API 调用的 trace 日志（输出请求和响应）
    trace_enabled: bool = Field(default=True, validation_alias="TRACE_ENABLED")
    # AgentService 本地 checkpoint / human review 状态库
    agent_state_db_path: str = Field(
        default="data/agent_state/state.sqlite3",
        validation_alias="AGENT_STATE_DB_PATH",
    )

    # 整个 Settings 类的读取规则
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
