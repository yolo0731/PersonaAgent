from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from agent_service.schemas import AgentReplyCommand, ChatRequest, no_reply_command


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


class HumanReviewRecord(BaseModel):
    thread_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    status: ReviewStatus
    request: ChatRequest
    edited_text: str | None = None
    created_at: str
    updated_at: str


class EditReviewRequest(BaseModel):
    edited_text: str = Field(min_length=1)


class ApproveReviewRequest(BaseModel):
    edited_text: str | None = Field(default=None, min_length=1)


class HumanReviewNotFoundError(KeyError):
    """Raised when a review thread is missing from the local store."""


def make_thread_id(request: ChatRequest) -> str:
    return f"conversation-{request.conversation_id}-message-{request.message_id}"


class HumanReviewStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def save_checkpoint(
        self,
        thread_id: str,
        state: Mapping[str, object],
        final_command: AgentReplyCommand | None = None,
    ) -> None:
        run_id = str(state["run_id"])
        now = _now()
        final_command_json = (
            _dump_json(final_command.model_dump(mode="json"))
            if final_command is not None
            else None
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (
                    thread_id, run_id, state_json, final_command_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    state_json = excluded.state_json,
                    final_command_json = COALESCE(
                        excluded.final_command_json,
                        checkpoints.final_command_json
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    thread_id,
                    run_id,
                    _dump_json(_jsonable(state)),
                    final_command_json,
                    now,
                ),
            )

    def load_checkpoint(self, thread_id: str) -> dict[str, object] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        state = json.loads(str(row["state_json"]))
        if not isinstance(state, dict):
            return None
        return state

    def create_pending(self, thread_id: str, state: Mapping[str, object]) -> HumanReviewRecord:
        self.save_checkpoint(thread_id, state)
        request = _request_from_state(state)
        now = _now()
        existing = self.get_review(thread_id)
        if existing is not None:
            return existing
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO human_reviews (
                    thread_id, run_id, status, request_json, state_json,
                    edited_text, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    thread_id,
                    request.run_id,
                    ReviewStatus.PENDING.value,
                    _dump_json(request.model_dump(mode="json")),
                    _dump_json(_jsonable(state)),
                    now,
                    now,
                ),
            )
        return self._require_review(thread_id)

    def get_review(self, thread_id: str) -> HumanReviewRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT thread_id, run_id, status, request_json, edited_text,
                       created_at, updated_at
                FROM human_reviews
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return _record_from_row(row)

    def edit(self, thread_id: str, edited_text: str) -> HumanReviewRecord:
        self._require_review(thread_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE human_reviews
                SET edited_text = ?, updated_at = ?
                WHERE thread_id = ? AND status != ?
                """,
                (edited_text, _now(), thread_id, ReviewStatus.COMPLETED.value),
            )
        return self._require_review(thread_id)

    def approve(self, thread_id: str, edited_text: str | None = None) -> HumanReviewRecord:
        self._require_review(thread_id)
        with self._connect() as conn:
            if edited_text is None:
                conn.execute(
                    """
                    UPDATE human_reviews
                    SET status = ?, updated_at = ?
                    WHERE thread_id = ? AND status != ?
                    """,
                    (
                        ReviewStatus.APPROVED.value,
                        _now(),
                        thread_id,
                        ReviewStatus.COMPLETED.value,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE human_reviews
                    SET status = ?, edited_text = ?, updated_at = ?
                    WHERE thread_id = ? AND status != ?
                    """,
                    (
                        ReviewStatus.APPROVED.value,
                        edited_text,
                        _now(),
                        thread_id,
                        ReviewStatus.COMPLETED.value,
                    ),
                )
        return self._require_review(thread_id)

    def reject(self, thread_id: str) -> HumanReviewRecord:
        self._require_review(thread_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE human_reviews
                SET status = ?, updated_at = ?
                WHERE thread_id = ? AND status != ?
                """,
                (
                    ReviewStatus.REJECTED.value,
                    _now(),
                    thread_id,
                    ReviewStatus.COMPLETED.value,
                ),
            )
        return self._require_review(thread_id)

    def mark_completed(
        self,
        thread_id: str,
        command: AgentReplyCommand,
    ) -> HumanReviewRecord:
        record = self._require_review(thread_id)
        checkpoint = self.load_checkpoint(thread_id) or {
            "run_id": record.run_id,
            "request": record.request.model_dump(mode="json"),
        }
        self.save_checkpoint(thread_id, checkpoint, command)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE human_reviews
                SET status = ?, updated_at = ?
                WHERE thread_id = ?
                """,
                (ReviewStatus.COMPLETED.value, _now(), thread_id),
            )
        return self._require_review(thread_id)

    def _require_review(self, thread_id: str) -> HumanReviewRecord:
        record = self.get_review(thread_id)
        if record is None:
            raise HumanReviewNotFoundError(thread_id)
        return record

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    final_command_json TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS human_reviews (
                    thread_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    edited_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn


def resume_human_review(thread_id: str, store: HumanReviewStore) -> AgentReplyCommand:
    record = store.get_review(thread_id)
    if record is None:
        raise HumanReviewNotFoundError(thread_id)

    if record.status == ReviewStatus.COMPLETED:
        return no_reply_command(record.request, "human_review_already_resumed")

    if record.status == ReviewStatus.PENDING:
        return no_reply_command(record.request, "human_review_pending")

    if record.status == ReviewStatus.REJECTED:
        command = no_reply_command(record.request, "human_review_rejected")
        store.mark_completed(thread_id, command)
        return command

    text = record.edited_text or f"mock reply: {record.request.text}"
    command = AgentReplyCommand(
        run_id=record.run_id,
        source_message_id=record.request.message_id,
        should_send=True,
        receiver_id=record.request.sender_id,
        text=text,
        client_message_id=f"pa-{record.run_id}",
        reason="human_review_approved",
    )
    store.mark_completed(thread_id, command)
    return command


def _record_from_row(row: sqlite3.Row) -> HumanReviewRecord:
    request = ChatRequest.model_validate(json.loads(str(row["request_json"])))
    edited_text_value = row["edited_text"]
    edited_text = str(edited_text_value) if edited_text_value is not None else None
    return HumanReviewRecord(
        thread_id=str(row["thread_id"]),
        run_id=str(row["run_id"]),
        status=ReviewStatus(str(row["status"])),
        request=request,
        edited_text=edited_text,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _request_from_state(state: Mapping[str, object]) -> ChatRequest:
    request = state["request"]
    if isinstance(request, ChatRequest):
        return request
    return ChatRequest.model_validate(request)


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _dump_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()
