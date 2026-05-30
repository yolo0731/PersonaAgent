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
    llm_request_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="LLM_REQUEST_TIMEOUT_SECONDS",
        gt=0.0,
    )
    dialogue_policy_mode: str = Field(default="rule", validation_alias="DIALOGUE_POLICY_MODE")
    dialogue_policy_max_retries: int = Field(
        default=2,
        validation_alias="DIALOGUE_POLICY_MAX_RETRIES",
        ge=1,
    )
    dialogue_policy_timeout_seconds: float | None = Field(
        default=None,
        validation_alias="DIALOGUE_POLICY_TIMEOUT_SECONDS",
        gt=0.0,
    )

    # DeepSeek 通过 OpenAI SDK 兼容协议接入，因此沿用 OPENAI_* 环境变量名。
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")

    # Embedding 模型和提供商。生产运行默认使用真实 provider；单元测试显式注入 mock。
    embedding_provider: str = Field(default="gemini", validation_alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="models/gemini-embedding-001",
        validation_alias="EMBEDDING_MODEL",
    )
    embedding_request_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="EMBEDDING_REQUEST_TIMEOUT_SECONDS",
        gt=0.0,
    )
    embedding_api_key: str | None = Field(default=None, validation_alias="EMBEDDING_API_KEY")
    embedding_base_url: str | None = Field(default=None, validation_alias="EMBEDDING_BASE_URL")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        validation_alias="GEMINI_BASE_URL",
    )
    # 向量数据库类型
    vector_db: str = Field(default="chroma", validation_alias="VECTOR_DB")
    chroma_path: str = Field(default="data/vector/chroma", validation_alias="CHROMA_PATH")
    memory_db_path: str = Field(
        default="data/state/memory/memory.sqlite3",
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
    # 授权 self_text -> target_reply 对话 pair 检索数量，用于让回复理解上下文语境。
    style_pair_top_k: int = Field(default=3, validation_alias="STYLE_PAIR_TOP_K")
    style_pairs_path: str = Field(
        default="data/authorized_style_records/processed/style_pairs.local.jsonl",
        validation_alias="STYLE_PAIRS_PATH",
    )
    style_samples_path: str = Field(
        default="data/authorized_style_records/processed/style_samples.local.jsonl",
        validation_alias="STYLE_SAMPLES_PATH",
    )
    # 指定当前 Bot persona 使用哪份授权风格；为空时沿用 sender_id 作为 persona_id。
    style_persona_id: str | None = Field(default=None, validation_alias="STYLE_PERSONA_ID")
    # 开启后，普通私聊 smalltalk 也会加载授权风格，适合真人授权风格 demo。
    style_on_smalltalk: bool = Field(default=False, validation_alias="STYLE_ON_SMALLTALK")
    # 开启后，普通私聊默认加载授权风格；知识/记忆类聊天也保持目标 persona 语气。
    style_on_private_chat: bool = Field(
        default=False,
        validation_alias="STYLE_ON_PRIVATE_CHAT",
    )
    # 本地授权风格资料，存放关系背景、昵称、语气规则等不会提交 Git 的私有信息。
    style_profile_path: str | None = Field(default=None, validation_alias="STYLE_PROFILE_PATH")
    # 普通聊天后自动保存对话记忆，后续可通过 Memory RAG 读取上下文。
    auto_memory_on_chat: bool = Field(default=True, validation_alias="AUTO_MEMORY_ON_CHAT")
    auto_memory_user_name: str = Field(default="演示用户", validation_alias="AUTO_MEMORY_USER_NAME")
    auto_memory_persona_name: str = Field(
        default="示例伙伴",
        validation_alias="AUTO_MEMORY_PERSONA_NAME",
    )
    # 安全通过后的 Agent 回复可追加为本地授权风格强化样本。
    style_reinforcement_enabled: bool = Field(
        default=False,
        validation_alias="STYLE_REINFORCEMENT_ENABLED",
    )
    style_reinforcement_samples_path: str = Field(
        default="data/authorized_style_records/processed/runtime_style_feedback.local.jsonl",
        validation_alias="STYLE_REINFORCEMENT_SAMPLES_PATH",
    )
    style_reinforcement_consent_id: str = Field(
        default="consent-demo-persona-runtime-style",
        validation_alias="STYLE_REINFORCEMENT_CONSENT_ID",
    )
    style_reinforcement_subject_user_id: int = Field(
        default=10001,
        validation_alias="STYLE_REINFORCEMENT_SUBJECT_USER_ID",
    )
    # Persona 和 prompt 模板配置
    persona_config_path: str = Field(
        default="agent_service/persona/persona.yaml",
        validation_alias="PERSONA_CONFIG_PATH",
    )

    # 是否开启 echo 模式（返回 LLM 原始输出）
    echo_mode: bool = Field(default=True, validation_alias="ECHO_MODE")
    # 是否开启 LLM API 调用的 trace 日志（输出请求和响应）
    trace_enabled: bool = Field(default=True, validation_alias="TRACE_ENABLED")
    # AgentService 本地 checkpoint / human review 状态库
    agent_state_db_path: str = Field(
        default="data/state/agent_state/state.sqlite3",
        validation_alias="AGENT_STATE_DB_PATH",
    )
    review_ui_token: str | None = Field(default=None, validation_alias="REVIEW_UI_TOKEN")

    # 整个 Settings 类的读取规则
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
