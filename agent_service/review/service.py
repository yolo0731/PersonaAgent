from __future__ import annotations

from agent_service.review.models import HumanReviewNotFoundError, ReviewStatus
from agent_service.review.store import HumanReviewStore
from agent_service.schemas import (
    AgentReplyCommand,
    ChatRequest,
    no_reply_command,
    send_reply_command,
)


def make_thread_id(request: ChatRequest) -> str:
    return f"conversation-{request.conversation_id}-message-{request.message_id}"


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

    command = send_reply_command(
        record.request,
        text=record.edited_text,
        reason="human_review_approved",
    )
    store.mark_completed(thread_id, command, action="resume")
    return command
