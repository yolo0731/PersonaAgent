from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _chat_payload(text: str = "我应该怎么用药？帮我给出具体剂量。") -> dict[str, object]:
    return {
        "run_id": "run-review",
        "conversation_type": 1,
        "conversation_id": 10011002,
        "message_id": 7001,
        "sender_id": 1002,
        "receiver_id": 1001,
        "text": text,
        "timestamp_ms": 1_700_000_001_000,
        "client_message_id": "alice-7001",
    }


def _client(db_path: Path) -> TestClient:
    from agent_service.config import Settings
    from agent_service.main import create_app

    return TestClient(
        create_app(Settings(_env_file=None, agent_state_db_path=str(db_path)))
    )


def test_thread_id_uses_conversation_and_incoming_message_id() -> None:
    from agent_service.review import make_thread_id
    from agent_service.schemas import ChatRequest

    request = ChatRequest.model_validate(_chat_payload())

    assert make_thread_id(request) == "conversation-10011002-message-7001"


def test_high_risk_chat_enters_pending_review_and_saves_checkpoint(tmp_path: Path) -> None:
    from agent_service.review import HumanReviewStore, ReviewStatus, make_thread_id
    from agent_service.schemas import ChatRequest

    db_path = tmp_path / "agent_state.sqlite3"
    client = _client(db_path)

    response = client.post("/chat", json=_chat_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["command"]["should_send"] is False
    assert body["command"]["reason"] == "human_review_pending"

    thread_id = make_thread_id(ChatRequest.model_validate(_chat_payload()))
    store = HumanReviewStore(db_path)
    pending = store.get_review(thread_id)
    assert pending is not None
    assert pending.status == ReviewStatus.PENDING
    assert pending.run_id == "run-review"
    checkpoint = store.load_checkpoint(thread_id)
    assert checkpoint is not None
    assert checkpoint["run_id"] == "run-review"


def test_approve_edit_and_resume_continues_with_same_thread_id(tmp_path: Path) -> None:
    from agent_service.review import HumanReviewStore, ReviewStatus, make_thread_id
    from agent_service.schemas import ChatRequest

    db_path = tmp_path / "agent_state.sqlite3"
    client = _client(db_path)
    thread_id = make_thread_id(ChatRequest.model_validate(_chat_payload()))

    client.post("/chat", json=_chat_payload())
    edit_response = client.post(
        f"/human-review/{thread_id}/edit",
        json={"edited_text": "approved edited reply"},
    )
    approve_response = client.post(f"/human-review/{thread_id}/approve")
    resume_response = client.post(f"/human-review/{thread_id}/resume")

    assert edit_response.status_code == 200
    assert edit_response.json()["status"] == ReviewStatus.PENDING
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == ReviewStatus.APPROVED
    assert resume_response.status_code == 200
    body = resume_response.json()
    assert body["ok"] is True
    assert body["command"]["run_id"] == "run-review"
    assert body["command"]["should_send"] is True
    assert body["command"]["receiver_id"] == 1002
    assert body["command"]["text"] == "approved edited reply"
    assert body["command"]["client_message_id"] == "pa-run-review"
    assert body["command"]["reason"] == "human_review_approved"

    completed = HumanReviewStore(db_path).get_review(thread_id)
    assert completed is not None
    assert completed.status == ReviewStatus.COMPLETED


def test_reject_and_resume_returns_noop_command(tmp_path: Path) -> None:
    from agent_service.review import ReviewStatus, make_thread_id
    from agent_service.schemas import ChatRequest

    db_path = tmp_path / "agent_state.sqlite3"
    client = _client(db_path)
    thread_id = make_thread_id(ChatRequest.model_validate(_chat_payload()))

    client.post("/chat", json=_chat_payload())
    reject_response = client.post(f"/human-review/{thread_id}/reject")
    resume_response = client.post(f"/human-review/{thread_id}/resume")

    assert reject_response.status_code == 200
    assert reject_response.json()["status"] == ReviewStatus.REJECTED
    assert resume_response.status_code == 200
    body = resume_response.json()
    assert body["ok"] is True
    assert body["command"]["should_send"] is False
    assert body["command"]["reason"] == "human_review_rejected"


def test_repeated_resume_does_not_generate_duplicate_send_command(tmp_path: Path) -> None:
    from agent_service.review import make_thread_id
    from agent_service.schemas import ChatRequest

    db_path = tmp_path / "agent_state.sqlite3"
    client = _client(db_path)
    thread_id = make_thread_id(ChatRequest.model_validate(_chat_payload()))

    client.post("/chat", json=_chat_payload())
    client.post(f"/human-review/{thread_id}/approve", json={"edited_text": "approved once"})

    first_resume = client.post(f"/human-review/{thread_id}/resume").json()
    second_resume = client.post(f"/human-review/{thread_id}/resume").json()

    assert first_resume["command"]["should_send"] is True
    assert first_resume["command"]["text"] == "approved once"
    assert second_resume["command"]["should_send"] is False
    assert second_resume["command"]["reason"] == "human_review_already_resumed"
