from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bot_client.protocol_parser import FriendProfile, IncomingMessage, ReceiptTraceEvent

FriendPolicyAction = Literal[
    "friend_list_synced",
    "accepted",
    "rejected",
    "left_pending",
    "accepted_push",
    "blocked_non_friend_message",
]
AgentReplyStatus = Literal["sent", "failed"]


@dataclass(frozen=True, slots=True)
class FriendPolicyTraceEvent:
    action: FriendPolicyAction
    user_id: int
    username: str
    reason: str


@dataclass(frozen=True, slots=True)
class AgentReplyTraceEvent:
    status: AgentReplyStatus
    dedup_key: str
    source_message_id: int
    receiver_id: int
    client_message_id: str | None
    reason: str


class JsonMessageState:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._processed_message_ids: set[int] = set()
        self._receipts: list[ReceiptTraceEvent] = []
        self._friends: dict[int, FriendProfile] = {}
        self._friend_policy_events: list[FriendPolicyTraceEvent] = []
        self._group_messages: list[IncomingMessage] = []
        self._agent_reply_events: list[AgentReplyTraceEvent] = []
        self._sent_agent_reply_dedup_keys: set[str] = set()
        self._load()

    @property
    def receipts(self) -> list[ReceiptTraceEvent]:
        return list(self._receipts)

    @property
    def friends(self) -> list[FriendProfile]:
        return list(self._friends.values())

    @property
    def friend_policy_events(self) -> list[FriendPolicyTraceEvent]:
        return list(self._friend_policy_events)

    @property
    def group_messages(self) -> list[IncomingMessage]:
        return list(self._group_messages)

    @property
    def agent_reply_events(self) -> list[AgentReplyTraceEvent]:
        return list(self._agent_reply_events)

    def has_processed(self, message_id: int) -> bool:
        return message_id in self._processed_message_ids

    def mark_processed(self, message_id: int) -> None:
        self._processed_message_ids.add(message_id)
        self._save()

    def record_receipt(self, event: ReceiptTraceEvent) -> None:
        self._receipts.append(event)
        self._save()

    def replace_friends(self, friends: list[FriendProfile]) -> None:
        self._friends = {friend.user_id: friend for friend in friends}
        self._save()

    def upsert_friend(self, friend: FriendProfile) -> None:
        self._friends[friend.user_id] = friend
        self._save()

    def is_friend(self, user_id: int) -> bool:
        return user_id in self._friends

    def record_friend_policy_event(self, event: FriendPolicyTraceEvent) -> None:
        self._friend_policy_events.append(event)
        self._save()

    def record_group_message(self, message: IncomingMessage) -> None:
        self._group_messages.append(message)
        self._save()

    def has_sent_agent_reply(self, dedup_key: str) -> bool:
        return dedup_key in self._sent_agent_reply_dedup_keys

    def record_agent_reply_event(self, event: AgentReplyTraceEvent) -> None:
        self._agent_reply_events.append(event)
        if event.status == "sent":
            self._sent_agent_reply_dedup_keys.add(event.dedup_key)
        self._save()

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        self._processed_message_ids = {
            int(value) for value in data.get("processed_message_ids", [])
        }
        self._receipts = [
            ReceiptTraceEvent(
                kind=item["kind"],
                message_id=int(item["message_id"]),
                conversation_id=int(item["conversation_id"]),
                peer_user_id=int(item["peer_user_id"]),
                delivery_status=int(item["delivery_status"]),
            )
            for item in data.get("receipts", [])
            if isinstance(item, dict)
        ]
        self._friends = {
            int(item["user_id"]): FriendProfile(
                user_id=int(item["user_id"]),
                username=str(item["username"]),
                nickname=str(item["nickname"]),
                online=bool(item["online"]),
            )
            for item in data.get("friends", [])
            if isinstance(item, dict)
        }
        self._friend_policy_events = [
            FriendPolicyTraceEvent(
                action=item["action"],
                user_id=int(item["user_id"]),
                username=str(item["username"]),
                reason=str(item["reason"]),
            )
            for item in data.get("friend_policy_events", [])
            if isinstance(item, dict)
        ]
        self._group_messages = [
            IncomingMessage(
                message_id=int(item["message_id"]),
                conversation_type=int(item["conversation_type"]),
                conversation_id=int(item["conversation_id"]),
                sender_id=int(item["sender_id"]),
                receiver_id=int(item["receiver_id"]),
                text=str(item["text"]),
                timestamp_ms=int(item["timestamp_ms"]),
                client_message_id=(
                    str(item["client_message_id"])
                    if item.get("client_message_id") is not None
                    else None
                ),
            )
            for item in data.get("group_messages", [])
            if isinstance(item, dict)
        ]
        self._agent_reply_events = [
            AgentReplyTraceEvent(
                status=item["status"],
                dedup_key=str(item["dedup_key"]),
                source_message_id=int(item["source_message_id"]),
                receiver_id=int(item["receiver_id"]),
                client_message_id=(
                    str(item["client_message_id"])
                    if item.get("client_message_id") is not None
                    else None
                ),
                reason=str(item["reason"]),
            )
            for item in data.get("agent_reply_events", [])
            if isinstance(item, dict)
        ]
        self._sent_agent_reply_dedup_keys = {
            event.dedup_key for event in self._agent_reply_events if event.status == "sent"
        }

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "processed_message_ids": sorted(self._processed_message_ids),
            "receipts": [
                {
                    "kind": event.kind,
                    "message_id": event.message_id,
                    "conversation_id": event.conversation_id,
                    "peer_user_id": event.peer_user_id,
                    "delivery_status": event.delivery_status,
                }
                for event in self._receipts
            ],
            "friends": [
                {
                    "user_id": friend.user_id,
                    "username": friend.username,
                    "nickname": friend.nickname,
                    "online": friend.online,
                }
                for friend in self._friends.values()
            ],
            "friend_policy_events": [
                {
                    "action": event.action,
                    "user_id": event.user_id,
                    "username": event.username,
                    "reason": event.reason,
                }
                for event in self._friend_policy_events
            ],
            "group_messages": [
                {
                    "message_id": message.message_id,
                    "conversation_type": message.conversation_type,
                    "conversation_id": message.conversation_id,
                    "sender_id": message.sender_id,
                    "receiver_id": message.receiver_id,
                    "text": message.text,
                    "timestamp_ms": message.timestamp_ms,
                    "client_message_id": message.client_message_id,
                }
                for message in self._group_messages
            ],
            "agent_reply_events": [
                {
                    "status": event.status,
                    "dedup_key": event.dedup_key,
                    "source_message_id": event.source_message_id,
                    "receiver_id": event.receiver_id,
                    "client_message_id": event.client_message_id,
                    "reason": event.reason,
                }
                for event in self._agent_reply_events
            ],
        }
        tmp_path = self._path.with_name(f"{self._path.name}.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self._path)
