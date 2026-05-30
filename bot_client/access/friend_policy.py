from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from bot_client.messages.state import (
    FriendPolicyAction,
    FriendPolicyTraceEvent,
    JsonMessageState,
)
from bot_client.protocol.codec import MessageType, Packet
from bot_client.protocol.parsers import (
    FRIEND_REQUEST_ACCEPTED,
    FRIEND_REQUEST_PENDING,
    FriendProfile,
    FriendRequest,
    parse_friend_action,
)
from bot_client.runtime.config import BotClientSettings


@dataclass(slots=True)
class FriendAccessPolicy:
    allowed_user_ids: set[int] = field(default_factory=set)
    allowed_usernames: set[str] = field(default_factory=set)
    auto_accept_friend_requests: bool = True
    reject_non_allowlisted_friend_requests: bool = True

    @classmethod
    def from_settings(cls, settings: BotClientSettings) -> FriendAccessPolicy:
        return cls(
            allowed_user_ids=_parse_user_ids(settings.allowed_user_ids),
            allowed_usernames=_parse_usernames(settings.allowed_usernames),
            auto_accept_friend_requests=settings.auto_accept_friend_requests,
            reject_non_allowlisted_friend_requests=(
                settings.reject_non_allowlisted_friend_requests
            ),
        )

    def allows(self, profile: FriendProfile) -> bool:
        return (
            profile.user_id in self.allowed_user_ids
            or profile.username in self.allowed_usernames
        )


class FriendPolicyClient(Protocol):
    async def list_friends(self) -> list[FriendProfile]: ...

    async def list_friend_requests(self) -> list[FriendRequest]: ...

    async def accept_friend_request(self, requester_id: int) -> FriendRequest: ...

    async def reject_friend_request(self, requester_id: int) -> FriendRequest: ...


class FriendPolicyHandler:
    def __init__(
        self,
        client: FriendPolicyClient,
        state: JsonMessageState,
        policy: FriendAccessPolicy,
    ) -> None:
        self._client = client
        self._state = state
        self._policy = policy

    async def sync_after_login(self) -> None:
        friends = await self._client.list_friends()
        self._state.replace_friends(friends)
        self._record(
            action="friend_list_synced",
            profile=None,
            reason=f"{len(friends)} friends",
        )

        for request in await self._client.list_friend_requests():
            if request.status != FRIEND_REQUEST_PENDING:
                continue
            await self._apply_request_policy(request)

    async def handle_packet(self, packet: Packet) -> None:
        if packet.header.msg_type != MessageType.FriendAcceptedPush:
            return
        accepted = parse_friend_action(packet)
        if accepted.status != FRIEND_REQUEST_ACCEPTED:
            return
        self._state.upsert_friend(accepted.profile)
        self._record(action="accepted_push", profile=accepted.profile, reason="server_push")

    async def _apply_request_policy(self, request: FriendRequest) -> None:
        if self._policy.allows(request.profile):
            if self._policy.auto_accept_friend_requests:
                accepted = await self._client.accept_friend_request(request.profile.user_id)
                self._state.upsert_friend(accepted.profile)
                self._record(action="accepted", profile=request.profile, reason="allowlist")
                return
            self._record(
                action="left_pending",
                profile=request.profile,
                reason="auto_accept_disabled",
            )
            return

        if self._policy.reject_non_allowlisted_friend_requests:
            await self._client.reject_friend_request(request.profile.user_id)
            self._record(action="rejected", profile=request.profile, reason="not_allowlisted")
            return

        self._record(action="left_pending", profile=request.profile, reason="not_allowlisted")

    def _record(
        self,
        *,
        action: FriendPolicyAction,
        profile: FriendProfile | None,
        reason: str,
    ) -> None:
        self._state.record_friend_policy_event(
            FriendPolicyTraceEvent(
                action=action,
                user_id=profile.user_id if profile is not None else 0,
                username=profile.username if profile is not None else "",
                reason=reason,
            )
        )


def _parse_user_ids(raw_value: str) -> set[int]:
    output: set[int] = set()
    for item in _split_csv(raw_value):
        value = int(item)
        if value <= 0:
            raise ValueError("allowed user ids must be positive")
        output.add(value)
    return output


def _parse_usernames(raw_value: str) -> set[str]:
    return set(_split_csv(raw_value))


def _split_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]
