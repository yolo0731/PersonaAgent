import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel


def test_settings_have_safe_defaults_without_env_file() -> None:
    from agent_service.config import Settings

    settings = Settings(_env_file=None)

    assert settings.agent_host == "127.0.0.1"
    assert settings.agent_port == 8088
    assert settings.llm_provider == "mock"
    assert settings.llm_request_timeout_seconds == 30.0
    assert settings.openai_api_key is None
    assert settings.openai_base_url is None
    assert settings.embedding_provider == "gemini"
    assert settings.embedding_model == "models/gemini-embedding-001"
    assert settings.embedding_request_timeout_seconds == 30.0
    assert settings.embedding_api_key is None
    assert settings.embedding_base_url is None
    assert settings.gemini_api_key is None
    assert settings.gemini_base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert settings.rag_top_k == 5
    assert settings.style_top_k == 8
    assert settings.style_profile_path is None
    assert settings.style_on_private_chat is False
    assert settings.auto_memory_on_chat is True
    assert settings.style_reinforcement_enabled is False
    assert (
        settings.style_reinforcement_samples_path
        == "data/authorized_style_records/processed/runtime_style_feedback.local.jsonl"
    )
    assert settings.echo_mode is True
    assert settings.trace_enabled is True


def test_settings_support_deepseek_openai_compatible_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_service.config import Settings

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "42.5")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "deepseek"
    assert settings.llm_model == "deepseek-v4-flash"
    assert settings.openai_api_key == "test-key"
    assert settings.openai_base_url == "https://api.deepseek.com"
    assert settings.llm_request_timeout_seconds == 42.5


def test_settings_support_runtime_authorized_style_persona(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_service.config import Settings

    monkeypatch.setenv("STYLE_PERSONA_ID", "demo_persona")
    monkeypatch.setenv("STYLE_ON_SMALLTALK", "true")
    monkeypatch.setenv("STYLE_ON_PRIVATE_CHAT", "true")
    monkeypatch.setenv(
        "STYLE_PROFILE_PATH",
        "data/authorized_style_records/processed/demo_persona.md",
    )
    monkeypatch.setenv("STYLE_REINFORCEMENT_ENABLED", "true")

    settings = Settings(_env_file=None)

    assert settings.style_persona_id == "demo_persona"
    assert settings.style_on_smalltalk is True
    assert settings.style_on_private_chat is True
    assert settings.style_profile_path == "data/authorized_style_records/processed/demo_persona.md"
    assert settings.style_reinforcement_enabled is True


def test_health_endpoint_returns_ok() -> None:
    from agent_service.main import create_app

    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "personaagent"}


class ReplySchema(BaseModel):
    should_reply: bool
    reply_text: str


class ReplyTextSchema(BaseModel):
    reply_text: str


@pytest.mark.asyncio
async def test_mock_llm_client_returns_fixed_structured_response() -> None:
    from agent_service.llm.base import LLMMessage
    from agent_service.llm.mock_client import MockLLMClient

    client = MockLLMClient(
        fixed_response={"should_reply": True, "reply_text": "mock reply"}
    )

    response = await client.generate(
        messages=[LLMMessage(role="user", content="hello")],
        response_model=ReplySchema,
    )

    assert response.model == "mock"
    assert response.content == "mock reply"
    assert response.structured == ReplySchema(should_reply=True, reply_text="mock reply")


def test_openai_client_can_be_constructed_without_api_key_for_unit_tests() -> None:
    from agent_service.llm.openai_client import OpenAILLMClient

    client = OpenAILLMClient(
        api_key=None,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )

    assert client.model == "deepseek-v4-flash"
    assert client.base_url == "https://api.deepseek.com"


@pytest.mark.asyncio
async def test_openai_client_maps_developer_role_for_compatible_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_service.llm.base import LLMMessage
    from agent_service.llm.openai_client import OpenAILLMClient

    captured: dict[str, object] = {}
    client_kwargs: dict[str, object] = {}

    class FakeMessage:
        content = '{"should_reply": true, "reply_text": "ok"}'

    class FakeChoice:
        message = FakeMessage()

    class FakeUsage:
        prompt_tokens = 3
        completion_tokens = 2

    class FakeCompletion:
        id = "completion-id"
        model = "deepseek-v4-flash"
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        async def create(self, **kwargs: object) -> FakeCompletion:
            captured.update(kwargs)
            return FakeCompletion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            client_kwargs.update(kwargs)
            self.chat = FakeChat()

    class FakeOpenAIModule:
        AsyncOpenAI = FakeAsyncOpenAI

    monkeypatch.setattr(
        "agent_service.llm.openai_client.import_module",
        lambda _name: FakeOpenAIModule,
    )

    client = OpenAILLMClient(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        timeout_seconds=42.5,
    )

    response = await client.generate(
        messages=[
            LLMMessage(role="system", content="system prompt"),
            LLMMessage(role="developer", content="developer prompt"),
            LLMMessage(role="user", content="hello"),
        ],
        response_model=ReplySchema,
    )

    assert response.structured == ReplySchema(should_reply=True, reply_text="ok")
    assert client_kwargs["timeout"] == 42.5
    assert captured["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "system", "content": "developer prompt"},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.asyncio
async def test_openai_client_wraps_plain_text_reply_when_schema_has_reply_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_service.llm.base import LLMMessage
    from agent_service.llm.openai_client import OpenAILLMClient

    class FakeMessage:
        content = "这是 DeepSeek 返回的普通文本回复。"

    class FakeChoice:
        message = FakeMessage()

    class FakeUsage:
        prompt_tokens = 4
        completion_tokens = 9

    class FakeCompletion:
        id = "completion-id"
        model = "deepseek-v4-flash"
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        async def create(self, **_kwargs: object) -> FakeCompletion:
            return FakeCompletion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = FakeChat()

    class FakeOpenAIModule:
        AsyncOpenAI = FakeAsyncOpenAI

    monkeypatch.setattr(
        "agent_service.llm.openai_client.import_module",
        lambda _name: FakeOpenAIModule,
    )

    client = OpenAILLMClient(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )

    response = await client.generate(
        messages=[LLMMessage(role="user", content="hello")],
        response_model=ReplyTextSchema,
    )

    assert response.content == "这是 DeepSeek 返回的普通文本回复。"
    assert response.structured == ReplyTextSchema(reply_text="这是 DeepSeek 返回的普通文本回复。")
