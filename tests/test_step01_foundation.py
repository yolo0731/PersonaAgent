import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel


def test_settings_have_safe_defaults_without_env_file() -> None:
    from agent_service.config import Settings

    settings = Settings(_env_file=None)

    assert settings.agent_host == "127.0.0.1"
    assert settings.agent_port == 8088
    assert settings.llm_provider == "mock"
    assert settings.openai_api_key is None
    assert settings.rag_top_k == 5
    assert settings.style_top_k == 8
    assert settings.echo_mode is True
    assert settings.trace_enabled is True


def test_health_endpoint_returns_ok() -> None:
    from agent_service.main import create_app

    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "personaagent"}


class ReplySchema(BaseModel):
    should_reply: bool
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

    client = OpenAILLMClient(api_key=None, model="example-model")

    assert client.model == "example-model"

