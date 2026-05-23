from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the PersonaAgent service."""

    service_name: str = "personaagent"
    agent_host: str = Field(default="127.0.0.1", validation_alias="AGENT_HOST")
    agent_port: int = Field(default=8088, validation_alias="AGENT_PORT")

    llm_provider: str = Field(default="mock", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="mock", validation_alias="LLM_MODEL")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    embedding_provider: str = Field(default="openai", validation_alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
    )
    vector_db: str = Field(default="chroma", validation_alias="VECTOR_DB")
    rag_top_k: int = Field(default=5, validation_alias="RAG_TOP_K")
    style_top_k: int = Field(default=8, validation_alias="STYLE_TOP_K")

    echo_mode: bool = Field(default=True, validation_alias="ECHO_MODE")
    trace_enabled: bool = Field(default=True, validation_alias="TRACE_ENABLED")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
