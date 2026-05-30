from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from agent_service.schemas import (
    AgentReplyCommand,
    ChatRequest,
    no_reply_command,
    send_reply_command,
)


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    EXPIRED = "expired"


class HumanReviewRecord(BaseModel):
    thread_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    status: ReviewStatus
    request: ChatRequest
    edited_text: str | None = None
    risk_reason: str | None = None
    expires_at: str | None = None
    created_at: str
    updated_at: str


class EditReviewRequest(BaseModel):
    edited_text: str = Field(min_length=1)
    operator: str = Field(default="local-admin", min_length=1)


class ApproveReviewRequest(BaseModel):
    edited_text: str | None = Field(default=None, min_length=1)
    operator: str = Field(default="local-admin", min_length=1)


class HumanReviewList(BaseModel):
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    items: list[HumanReviewRecord]


class HumanReviewAuditEntry(BaseModel):
    audit_id: int = Field(ge=1)
    thread_id: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    action: str = Field(min_length=1)
    before_status: ReviewStatus | None = None
    after_status: ReviewStatus | None = None
    edited_text: str | None = None
    final_command: AgentReplyCommand | None = None
    created_at: str


class HumanReviewDetail(BaseModel):
    record: HumanReviewRecord
    checkpoint_status: str
    final_command: AgentReplyCommand | None = None
    agent_draft: str = ""
    retrieved_context: list[str] = Field(default_factory=list)
    tool_results: list[str] = Field(default_factory=list)
    trace_summary: list[str] = Field(default_factory=list)
    audit_log: list[HumanReviewAuditEntry] = Field(default_factory=list)


class HumanReviewNotFoundError(KeyError):
    """Raised when a review thread is missing from the local store."""


class HumanReviewInvalidTransitionError(ValueError):
    """Raised when a review mutation would not change the current state."""


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

    def load_final_command(self, thread_id: str) -> AgentReplyCommand | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT final_command_json FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None or row["final_command_json"] is None:
            return None
        return AgentReplyCommand.model_validate(json.loads(str(row["final_command_json"])))

    def create_pending(self, thread_id: str, state: Mapping[str, object]) -> HumanReviewRecord:
        self.save_checkpoint(thread_id, state)
        request = _request_from_state(state)
        now = _now()
        existing = self.get_review(thread_id)
        if existing is not None:
            return existing
        risk_reason = _risk_reason_from_state(state)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO human_reviews (
                    thread_id, run_id, status, request_json, state_json,
                    edited_text, risk_reason, expires_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?)
                """,
                (
                    thread_id,
                    request.run_id,
                    ReviewStatus.PENDING.value,
                    _dump_json(request.model_dump(mode="json")),
                    _dump_json(_jsonable(state)),
                    risk_reason,
                    now,
                    now,
                ),
            )
        record = self._require_review(thread_id)
        self._append_audit(
            thread_id,
            action="create",
            operator="system",
            before_status=None,
            after_status=record.status,
        )
        return record

    def get_review(self, thread_id: str) -> HumanReviewRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT thread_id, run_id, status, request_json, edited_text,
                       risk_reason, expires_at,
                       created_at, updated_at
                FROM human_reviews
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return _record_from_row(row)

    def list_reviews(
        self,
        *,
        status: ReviewStatus | None = None,
        keyword: str | None = None,
        risk_reason: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> HumanReviewList:
        bounded_limit = max(1, min(limit, 200))
        bounded_offset = max(0, offset)
        where, params = _review_filters(
            status=status,
            keyword=keyword,
            risk_reason=risk_reason,
        )
        with self._connect() as conn:
            count_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM human_reviews {where}",
                params,
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT thread_id, run_id, status, request_json, edited_text,
                       risk_reason, expires_at, created_at, updated_at
                FROM human_reviews
                {where}
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (*params, bounded_limit, bounded_offset),
            ).fetchall()
        return HumanReviewList(
            total=int(count_row["total"] if count_row is not None else 0),
            limit=bounded_limit,
            offset=bounded_offset,
            items=[_record_from_row(row) for row in rows],
        )

    def detail(self, thread_id: str) -> HumanReviewDetail:
        record = self._require_review(thread_id)
        checkpoint = self.load_checkpoint(thread_id)
        final_command = self.load_final_command(thread_id)
        return HumanReviewDetail(
            record=record,
            checkpoint_status="saved" if checkpoint is not None else "missing",
            final_command=final_command,
            agent_draft=_string_from_state(checkpoint, "draft"),
            retrieved_context=_string_list_from_state(checkpoint, "retrieved_context"),
            tool_results=_string_list_from_state(checkpoint, "tool_results"),
            trace_summary=_trace_summary_from_state(checkpoint),
            audit_log=self.audit_log(thread_id),
        )

    def audit_log(self, thread_id: str) -> list[HumanReviewAuditEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT audit_id, thread_id, operator, action, before_status,
                       after_status, edited_text, final_command_json, created_at
                FROM human_review_audit
                WHERE thread_id = ?
                ORDER BY audit_id ASC
                """,
                (thread_id,),
            ).fetchall()
        return [_audit_from_row(row) for row in rows]

    def edit(
        self,
        thread_id: str,
        edited_text: str,
        *,
        operator: str = "local-admin",
    ) -> HumanReviewRecord:
        before = self._require_review(thread_id)
        _ensure_mutable_review(before, action="edit")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE human_reviews
                SET edited_text = ?, updated_at = ?
                WHERE thread_id = ? AND status != ?
                """,
                (edited_text, _now(), thread_id, ReviewStatus.COMPLETED.value),
            )
        after = self._require_review(thread_id)
        self._append_audit(
            thread_id,
            action="edit",
            operator=operator,
            before_status=before.status,
            after_status=after.status,
            edited_text=edited_text,
        )
        return after

    def approve(
        self,
        thread_id: str,
        edited_text: str | None = None,
        *,
        operator: str = "local-admin",
    ) -> HumanReviewRecord:
        before = self._require_review(thread_id)
        _ensure_mutable_review(before, action="approve")
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
        after = self._require_review(thread_id)
        self._append_audit(
            thread_id,
            action="approve",
            operator=operator,
            before_status=before.status,
            after_status=after.status,
            edited_text=edited_text,
        )
        return after

    def reject(
        self,
        thread_id: str,
        *,
        operator: str = "local-admin",
    ) -> HumanReviewRecord:
        before = self._require_review(thread_id)
        _ensure_mutable_review(before, action="reject")
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
        after = self._require_review(thread_id)
        self._append_audit(
            thread_id,
            action="reject",
            operator=operator,
            before_status=before.status,
            after_status=after.status,
        )
        return after

    def mark_completed(
        self,
        thread_id: str,
        command: AgentReplyCommand,
        *,
        operator: str = "system",
        action: str = "complete",
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
        after = self._require_review(thread_id)
        self._append_audit(
            thread_id,
            action=action,
            operator=operator,
            before_status=record.status,
            after_status=after.status,
            final_command=command,
        )
        return after

    def expire_pending_before(self, deadline: str) -> int:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT thread_id, status
                FROM human_reviews
                WHERE status = ? AND expires_at IS NOT NULL AND expires_at < ?
                """,
                (ReviewStatus.PENDING.value, deadline),
            ).fetchall()
            conn.execute(
                """
                UPDATE human_reviews
                SET status = ?, updated_at = ?
                WHERE status = ? AND expires_at IS NOT NULL AND expires_at < ?
                """,
                (
                    ReviewStatus.EXPIRED.value,
                    _now(),
                    ReviewStatus.PENDING.value,
                    deadline,
                ),
            )
        for row in rows:
            self._append_audit(
                str(row["thread_id"]),
                action="expire",
                operator="system",
                before_status=ReviewStatus(str(row["status"])),
                after_status=ReviewStatus.EXPIRED,
            )
        return len(rows)

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
                    risk_reason TEXT,
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS human_review_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    operator TEXT NOT NULL,
                    action TEXT NOT NULL,
                    before_status TEXT,
                    after_status TEXT,
                    edited_text TEXT,
                    final_command_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            _ensure_columns(
                conn,
                table="human_reviews",
                columns={
                    "risk_reason": "TEXT",
                    "expires_at": "TEXT",
                },
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _append_audit(
        self,
        thread_id: str,
        *,
        action: str,
        operator: str,
        before_status: ReviewStatus | None,
        after_status: ReviewStatus | None,
        edited_text: str | None = None,
        final_command: AgentReplyCommand | None = None,
    ) -> None:
        final_command_json = (
            _dump_json(final_command.model_dump(mode="json"))
            if final_command is not None
            else None
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO human_review_audit (
                    thread_id, operator, action, before_status, after_status,
                    edited_text, final_command_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    operator,
                    action,
                    before_status.value if before_status is not None else None,
                    after_status.value if after_status is not None else None,
                    edited_text,
                    final_command_json,
                    _now(),
                ),
            )


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
        store.mark_completed(thread_id, command, action="resume")
        return command

    if record.edited_text is None:
        command = no_reply_command(record.request, "human_review_missing_edit")
        store.mark_completed(thread_id, command, action="resume")
        return command

    text = record.edited_text
    command = send_reply_command(
        record.request,
        text=text,
        reason="human_review_approved",
    )
    store.mark_completed(thread_id, command, action="resume")
    return command


def _record_from_row(row: sqlite3.Row) -> HumanReviewRecord:
    request = ChatRequest.model_validate(json.loads(str(row["request_json"])))
    edited_text_value = row["edited_text"]
    edited_text = str(edited_text_value) if edited_text_value is not None else None
    risk_reason_value = row["risk_reason"]
    expires_at_value = row["expires_at"]
    return HumanReviewRecord(
        thread_id=str(row["thread_id"]),
        run_id=str(row["run_id"]),
        status=ReviewStatus(str(row["status"])),
        request=request,
        edited_text=edited_text,
        risk_reason=str(risk_reason_value) if risk_reason_value is not None else None,
        expires_at=str(expires_at_value) if expires_at_value is not None else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _ensure_mutable_review(record: HumanReviewRecord, *, action: str) -> None:
    if record.status == ReviewStatus.COMPLETED:
        raise HumanReviewInvalidTransitionError(
            f"cannot {action} completed review {record.thread_id}"
        )


def _audit_from_row(row: sqlite3.Row) -> HumanReviewAuditEntry:
    final_command_json = row["final_command_json"]
    final_command = (
        AgentReplyCommand.model_validate(json.loads(str(final_command_json)))
        if final_command_json is not None
        else None
    )
    before_status = row["before_status"]
    after_status = row["after_status"]
    return HumanReviewAuditEntry(
        audit_id=int(row["audit_id"]),
        thread_id=str(row["thread_id"]),
        operator=str(row["operator"]),
        action=str(row["action"]),
        before_status=ReviewStatus(str(before_status)) if before_status is not None else None,
        after_status=ReviewStatus(str(after_status)) if after_status is not None else None,
        edited_text=str(row["edited_text"]) if row["edited_text"] is not None else None,
        final_command=final_command,
        created_at=str(row["created_at"]),
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


def _ensure_columns(
    conn: sqlite3.Connection,
    *,
    table: str,
    columns: Mapping[str, str],
) -> None:
    existing = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, declaration in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def _review_filters(
    *,
    status: ReviewStatus | None,
    keyword: str | None,
    risk_reason: str | None,
) -> tuple[str, tuple[object, ...]]:
    clauses: list[str] = []
    params: list[object] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status.value)
    if keyword:
        pattern = f"%{keyword}%"
        clauses.append(
            """
            (
                thread_id LIKE ?
                OR run_id LIKE ?
                OR request_json LIKE ?
                OR COALESCE(edited_text, '') LIKE ?
            )
            """
        )
        params.extend([pattern, pattern, pattern, pattern])
    if risk_reason:
        clauses.append("COALESCE(risk_reason, '') LIKE ?")
        params.append(f"%{risk_reason}%")
    if not clauses:
        return "", tuple(params)
    return "WHERE " + " AND ".join(clauses), tuple(params)


def _risk_reason_from_state(state: Mapping[str, object]) -> str | None:
    safety = state.get("safety_result")
    if isinstance(safety, BaseModel):
        reason = getattr(safety, "reason", None)
        if reason:
            return str(reason)
    if isinstance(safety, Mapping):
        reason = safety.get("reason")
        if reason:
            return str(reason)
    decision = state.get("decision")
    if isinstance(decision, BaseModel):
        need_review = getattr(decision, "need_human_review", False)
        reason = getattr(decision, "reason", None)
        if need_review and reason:
            return str(reason)
    if isinstance(decision, Mapping) and decision.get("need_human_review"):
        reason = decision.get("reason")
        if reason:
            return str(reason)
    return None


def _string_list_from_state(
    state: Mapping[str, object] | None,
    key: str,
) -> list[str]:
    if state is None:
        return []
    value = state.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _string_from_state(
    state: Mapping[str, object] | None,
    key: str,
) -> str:
    if state is None:
        return ""
    value = state.get(key)
    return str(value) if value is not None else ""


def _trace_summary_from_state(state: Mapping[str, object] | None) -> list[str]:
    if state is None:
        return []
    final_command = state.get("final_command")
    if isinstance(final_command, Mapping):
        trace_summary = final_command.get("trace_summary")
        if isinstance(trace_summary, list):
            return [str(item) for item in trace_summary]
    trace = state.get("trace")
    if not isinstance(trace, list):
        return []
    summary: list[str] = []
    for item in trace:
        if isinstance(item, Mapping):
            node = item.get("node")
            action = item.get("action")
            if node is not None and action is not None:
                summary.append(f"{node}:{action}")
    return summary
