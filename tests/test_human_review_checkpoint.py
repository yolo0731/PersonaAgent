from __future__ import annotations

import csv
import io
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
        create_app(
            Settings(
                _env_file=None,
                embedding_provider="mock",
                agent_state_db_path=str(db_path),
                memory_db_path=str(db_path.parent / "memory.sqlite3"),
                chroma_path=str(db_path.parent / "chroma"),
                knowledge_docs_path=str(db_path.parent / "knowledge_docs"),
                style_samples_path=str(db_path.parent / "style_samples.local.jsonl"),
                style_pairs_path=str(db_path.parent / "style_pairs.local.jsonl"),
            )
        )
    )


def _token_client(db_path: Path, token: str) -> TestClient:
    from agent_service.config import Settings
    from agent_service.main import create_app

    return TestClient(
        create_app(
            Settings(
                _env_file=None,
                embedding_provider="mock",
                agent_state_db_path=str(db_path),
                memory_db_path=str(db_path.parent / "memory.sqlite3"),
                chroma_path=str(db_path.parent / "chroma"),
                knowledge_docs_path=str(db_path.parent / "knowledge_docs"),
                style_samples_path=str(db_path.parent / "style_samples.local.jsonl"),
                style_pairs_path=str(db_path.parent / "style_pairs.local.jsonl"),
                review_ui_token=token,
            )
        )
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


def test_approve_without_edit_does_not_send_mock_reply(tmp_path: Path) -> None:
    from agent_service.review import make_thread_id
    from agent_service.schemas import ChatRequest

    db_path = tmp_path / "agent_state.sqlite3"
    client = _client(db_path)
    thread_id = make_thread_id(ChatRequest.model_validate(_chat_payload()))

    client.post("/chat", json=_chat_payload())
    approve_response = client.post(f"/human-review/{thread_id}/approve")
    resume_response = client.post(f"/human-review/{thread_id}/resume")

    assert approve_response.status_code == 200
    body = resume_response.json()
    assert body["ok"] is True
    assert body["command"]["should_send"] is False
    assert body["command"]["text"] == ""
    assert body["command"]["reason"] == "human_review_missing_edit"


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


def test_review_list_detail_audit_and_export_routes(tmp_path: Path) -> None:
    from agent_service.review import make_thread_id
    from agent_service.schemas import ChatRequest

    db_path = tmp_path / "agent_state.sqlite3"
    client = _client(db_path)
    payload = _chat_payload("我应该怎么用药？帮我给出具体剂量。")
    thread_id = make_thread_id(ChatRequest.model_validate(payload))

    client.post("/chat", json=payload)
    client.post(
        f"/human-review/{thread_id}/edit",
        json={"edited_text": "请咨询医生，我不能给出剂量。", "operator": "reviewer-a"},
    )

    list_response = client.get(
        "/human-review",
        params={
            "status": "pending",
            "q": "用药",
            "risk_reason": "medical",
            "limit": 10,
            "offset": 0,
        },
    )
    detail_response = client.get(f"/human-review/{thread_id}")
    json_export = client.get("/human-review/export", params={"format": "json"})
    csv_export = client.get("/human-review/export", params={"format": "csv"})

    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["total"] == 1
    assert listed["items"][0]["thread_id"] == thread_id
    assert listed["items"][0]["risk_reason"] == "high_risk_domain:medical"

    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["record"]["thread_id"] == thread_id
    assert detail["record"]["status"] == "pending"
    assert detail["record"]["risk_reason"] == "high_risk_domain:medical"
    assert detail["checkpoint_status"] == "saved"
    assert detail["trace_summary"][0].startswith("dialogue_policy:")
    assert detail["final_command"]["reason"] == "human_review_pending"
    assert [entry["action"] for entry in detail["audit_log"]] == ["create", "edit"]
    assert detail["audit_log"][-1]["operator"] == "reviewer-a"

    assert json_export.status_code == 200
    assert json_export.json()["items"][0]["thread_id"] == thread_id

    assert csv_export.status_code == 200
    rows = list(csv.DictReader(io.StringIO(csv_export.text)))
    assert rows[0]["thread_id"] == thread_id
    assert rows[0]["risk_reason"] == "high_risk_domain:medical"


def test_review_ui_routes_return_operational_html(tmp_path: Path) -> None:
    from agent_service.review import make_thread_id
    from agent_service.schemas import ChatRequest

    db_path = tmp_path / "agent_state.sqlite3"
    client = _client(db_path)
    payload = _chat_payload("我应该怎么用药？帮我给出具体剂量。")
    thread_id = make_thread_id(ChatRequest.model_validate(payload))
    client.post("/chat", json=payload)

    list_page = client.get("/human-review/ui")
    detail_page = client.get(f"/human-review/ui/{thread_id}")

    assert list_page.status_code == 200
    assert "Human Review" in list_page.text
    assert "pending" in list_page.text
    assert "/human-review/export?format=csv" in list_page.text

    assert detail_page.status_code == 200
    assert thread_id in detail_page.text
    assert "Agent Draft" in detail_page.text
    assert "edit-approve-resume" in detail_page.text


def test_review_routes_require_bearer_token_when_configured(tmp_path: Path) -> None:
    db_path = tmp_path / "agent_state.sqlite3"
    client = _token_client(db_path, token="secret-review-token")

    unauthorized_api = client.get("/human-review")
    unauthorized_ui = client.get("/human-review/ui")
    authorized_api = client.get(
        "/human-review",
        headers={"Authorization": "Bearer secret-review-token"},
    )

    assert unauthorized_api.status_code == 401
    assert unauthorized_ui.status_code == 401
    assert authorized_api.status_code == 200


def test_review_ui_token_bootstrap_allows_browser_actions(tmp_path: Path) -> None:
    from agent_service.review import make_thread_id
    from agent_service.schemas import ChatRequest

    db_path = tmp_path / "agent_state.sqlite3"
    client = _token_client(db_path, token="secret-review-token")
    thread_id = make_thread_id(ChatRequest.model_validate(_chat_payload()))

    client.post("/chat", json=_chat_payload())

    list_page = client.get("/human-review/ui?token=secret-review-token")
    detail_page = client.get(f"/human-review/ui/{thread_id}?token=secret-review-token")

    assert list_page.status_code == 200
    assert f"/human-review/ui/{thread_id}?token=secret-review-token" in list_page.text
    assert "localStorage" in list_page.text
    assert "Authorization" in detail_page.text
    assert "Bearer" in detail_page.text


def test_review_ui_filter_and_pagination_preserve_token(tmp_path: Path) -> None:
    db_path = tmp_path / "agent_state.sqlite3"
    client = _token_client(db_path, token="secret-review-token")
    first = _chat_payload()
    second = _chat_payload()
    second["message_id"] = 7002
    second["client_message_id"] = "alice-7002"

    client.post("/chat", json=first)
    client.post("/chat", json=second)

    page = client.get(
        "/human-review/ui",
        params={
            "token": "secret-review-token",
            "status": "pending",
            "q": "用药",
            "risk_reason": "medical",
            "limit": 1,
            "offset": 0,
        },
    )

    assert page.status_code == 200
    assert 'name="token" value="secret-review-token"' in page.text
    assert "/human-review/ui?status=pending" in page.text
    assert "q=%E7%94%A8%E8%8D%AF" in page.text
    assert "risk_reason=medical" in page.text
    assert "limit=1" in page.text
    assert "offset=1" in page.text
    assert "token=secret-review-token" in page.text


def test_completed_review_mutations_are_rejected_and_not_audited(tmp_path: Path) -> None:
    from agent_service.review import HumanReviewStore, make_thread_id
    from agent_service.schemas import ChatRequest

    db_path = tmp_path / "agent_state.sqlite3"
    client = _client(db_path)
    thread_id = make_thread_id(ChatRequest.model_validate(_chat_payload()))

    client.post("/chat", json=_chat_payload())
    client.post(f"/human-review/{thread_id}/approve", json={"edited_text": "approved once"})
    client.post(f"/human-review/{thread_id}/resume")
    store = HumanReviewStore(db_path)
    audit_count = len(store.audit_log(thread_id))

    edit_response = client.post(
        f"/human-review/{thread_id}/edit",
        json={"edited_text": "should not be recorded"},
    )
    approve_response = client.post(
        f"/human-review/{thread_id}/approve",
        json={"edited_text": "should not be recorded"},
    )
    reject_response = client.post(f"/human-review/{thread_id}/reject")

    assert edit_response.status_code == 409
    assert approve_response.status_code == 409
    assert reject_response.status_code == 409
    assert len(HumanReviewStore(db_path).audit_log(thread_id)) == audit_count
