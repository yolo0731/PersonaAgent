from __future__ import annotations


def test_settings_support_gemini_embedding_provider(
    monkeypatch,
) -> None:
    from agent_service.config import Settings

    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

    settings = Settings(_env_file=None)

    assert settings.embedding_provider == "gemini"
    assert settings.embedding_model == "models/gemini-embedding-001"
    assert settings.gemini_api_key == "test-gemini-key"
    assert settings.gemini_base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_gemini_embedding_client_maps_embed_content_response(monkeypatch) -> None:
    from agent_service.rag.embeddings import GeminiEmbeddingClient

    captured: list[dict[str, object]] = []

    def fake_request_json(
        method: str,
        url: str,
        *,
        payload: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> object:
        captured.append(
            {
                "method": method,
                "url": url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"embedding": {"values": [0.25, -0.5, 0.75]}}

    monkeypatch.setattr("agent_service.rag.embeddings._request_json", fake_request_json)

    client = GeminiEmbeddingClient(
        api_key="test-gemini-key",
        model="models/gemini-embedding-001",
        timeout_seconds=12.5,
    )

    assert client.embed_query("hello embeddings") == [0.25, -0.5, 0.75]
    assert captured == [
        {
            "method": "POST",
            "url": (
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-embedding-001:embedContent?key=test-gemini-key"
            ),
            "payload": {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "hello embeddings"}]},
            },
            "timeout_seconds": 12.5,
        }
    ]


def test_container_builds_configured_embedding_client_and_scoped_collection_names() -> None:
    from agent_service.config import Settings
    from agent_service.container import build_embedding_client, embedding_collection_name
    from agent_service.rag.embeddings import GeminiEmbeddingClient

    settings = Settings(
        _env_file=None,
        embedding_provider="gemini",
        embedding_model="models/gemini-embedding-001",
        gemini_api_key="test-gemini-key",
    )

    client = build_embedding_client(settings)

    assert isinstance(client, GeminiEmbeddingClient)
    assert embedding_collection_name("knowledge", settings) == (
        "knowledge_gemini_models_gemini_embedding_001"
    )
    assert embedding_collection_name("memory", settings) == (
        "memory_gemini_models_gemini_embedding_001"
    )


def test_default_embedding_client_is_real_provider_not_mock() -> None:
    from agent_service.config import Settings
    from agent_service.container import build_embedding_client
    from agent_service.rag.embeddings import GeminiEmbeddingClient, MockEmbeddingClient

    client = build_embedding_client(Settings(_env_file=None))

    assert isinstance(client, GeminiEmbeddingClient)
    assert not isinstance(client, MockEmbeddingClient)
