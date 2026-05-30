from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Protocol, cast

from bot_client.access.friend_policy import (
    FriendAccessPolicy,
    FriendPolicyClient,
    FriendPolicyHandler,
)
from bot_client.agent.api import AgentApiClient, AgentCommandClient, AgentServiceMessageProcessor
from bot_client.connection.client import BotClient, BotIdentity
from bot_client.messages.echo import EchoMessageProcessor
from bot_client.messages.handler import BotMessageHandler, ReliabilityClient
from bot_client.messages.state import JsonMessageState
from bot_client.protocol.codec import MessageType, Packet
from bot_client.protocol.parsers import IncomingMessage
from bot_client.runtime.config import BotClientSettings


class EchoRuntimeClient(FriendPolicyClient, ReliabilityClient, Protocol):
    push_queue: asyncio.Queue[Packet]

    async def connect(self) -> None: ...

    async def login(self) -> BotIdentity: ...

    async def close(self) -> None: ...

class AgentRuntimeClient(FriendPolicyClient, ReliabilityClient, Protocol):
    push_queue: asyncio.Queue[Packet]

    async def connect(self) -> None: ...

    async def login(self) -> BotIdentity: ...

    async def close(self) -> None: ...

    async def pull_history_messages(
        self,
        *,
        conversation_type: int,
        conversation_id: int,
        before_message_id: int = 0,
        limit: int = 8,
    ) -> list[IncomingMessage]: ...

# BotClient 的 echo 运行时协调器。它组合 BotClient、好友策略、消息可靠性处理和本地状态
class EchoBotRuntime:
    def __init__(
        self,
        settings: BotClientSettings,
        *,
        client: EchoRuntimeClient | None = None,
        state_path: str | Path | None = None,
        state: JsonMessageState | None = None,
    ) -> None:
        self._settings = settings
        self._client: EchoRuntimeClient = (
            client if client is not None else cast(EchoRuntimeClient, BotClient(settings))
        )
        self._state = state or JsonMessageState(state_path or settings.bot_state_path)
        self._friend_handler = FriendPolicyHandler(
            client=self._client,
            state=self._state,
            policy=FriendAccessPolicy.from_settings(settings),
        )
        self._message_handler = BotMessageHandler(
            client=self._client,
            state=self._state,
            processor=EchoMessageProcessor(enabled=settings.echo_mode),
            require_friendship=True,
        )
        self._push_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> JsonMessageState:
        return self._state

    @property
    def running(self) -> bool:
        return self._push_task is not None and not self._push_task.done()

    # 负责分发 push：好友 accepted push 交给好友策略，其他消息交给消息处理器
    async def start(self) -> None:
        if self.running:
            return
        await self._client.connect()
        await self._client.login()
        await self._friend_handler.sync_after_login()
        await self._message_handler.sync_offline_after_login(
            self._settings.offline_message_limit
        )
        self._push_task = asyncio.create_task(self._push_loop())

    async def stop(self) -> None:
        task = self._push_task
        self._push_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._client.close()

    async def handle_packet(self, packet: Packet) -> None:
        if packet.header.msg_type == MessageType.FriendAcceptedPush:
            await self._friend_handler.handle_packet(packet)
            return
        await self._message_handler.handle_packet(packet)

    async def _push_loop(self) -> None:
        while True:
            packet = await self._client.push_queue.get()
            await self.handle_packet(packet)

class AgentBotRuntime:
    def __init__(
        self,
        settings: BotClientSettings,
        *,
        client: AgentRuntimeClient | None = None,
        agent_client: AgentCommandClient | None = None,
        state_path: str | Path | None = None,
        state: JsonMessageState | None = None,
    ) -> None:
        self._settings = settings
        self._client: AgentRuntimeClient = (
            client if client is not None else cast(AgentRuntimeClient, BotClient(settings))
        )
        self._state = state or JsonMessageState(state_path or settings.bot_state_path)
        self._friend_handler = FriendPolicyHandler(
            client=self._client,
            state=self._state,
            policy=FriendAccessPolicy.from_settings(settings),
        )
        command_client = agent_client or AgentApiClient.from_settings(settings)
        self._message_handler = BotMessageHandler(
            client=self._client,
            state=self._state,
            processor=AgentServiceMessageProcessor(
                command_client,
                history_loader=self._client.pull_history_messages,
                recent_context_limit=settings.recent_context_limit,
            ),
            require_friendship=True,
        )
        self._push_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> JsonMessageState:
        return self._state

    @property
    def running(self) -> bool:
        return self._push_task is not None and not self._push_task.done()

    async def start(self) -> None:
        if self.running:
            return
        await self._client.connect()
        await self._client.login()
        await self._friend_handler.sync_after_login()
        await self._message_handler.sync_offline_after_login(
            self._settings.offline_message_limit
        )
        self._push_task = asyncio.create_task(self._push_loop())

    async def stop(self) -> None:
        task = self._push_task
        self._push_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._client.close()

    async def handle_packet(self, packet: Packet) -> None:
        if packet.header.msg_type == MessageType.FriendAcceptedPush:
            await self._friend_handler.handle_packet(packet)
            return
        await self._message_handler.handle_packet(packet)

# 等待push_queue中的包，处理完后继续等待，直到被取消
    async def _push_loop(self) -> None:
        while True:
            packet = await self._client.push_queue.get()
            await self.handle_packet(packet)
