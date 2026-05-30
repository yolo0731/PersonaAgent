from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent_service.schemas import AgentReplyCommand
from bot_client.connection.client import BotIdentity
from bot_client.protocol.codec import (
    MessageType,
    Packet,
    PacketHeader,
    TlvType,
    append_string,
    append_uint64,
)


def _packet(msg_type: MessageType, body: bytes) -> Packet:
    return Packet(header=PacketHeader(msg_type=msg_type, seq_id=99), body=body)


def _private_message_body(
    *,
    message_id: int = 7401,
    sender_id: int = 1002,
    receiver_id: int = 1001,
    text: str = "hello agent",
) -> bytes:
    body = bytearray()
    append_uint64(TlvType.MessageId, message_id, body)
    append_uint64(TlvType.ConversationType, 1, body)
    append_uint64(TlvType.ConversationId, 10011002, body)
    append_uint64(TlvType.SenderId, sender_id, body)
    append_uint64(TlvType.ReceiverId, receiver_id, body)
    append_string(TlvType.MessageText, text, body)
    append_string(TlvType.ClientMessageId, f"alice-{message_id}", body)
    append_uint64(TlvType.TimestampMs, 1_700_000_001_000, body)
    return bytes(body)


def _reply_command(
    *,
    run_id: str = "run-agent-send",
    source_message_id: int = 7401,
    should_send: bool = True,
    text: str = "agent reply",
    dedup_key: str | None = None,
) -> AgentReplyCommand:
    return AgentReplyCommand(
        run_id=run_id,
        source_message_id=source_message_id,
        should_send=should_send,
        receiver_id=1002,
        conversation_type=1,
        conversation_id=10011002,
        text=text if should_send else "",
        client_message_id=f"pa-{run_id}" if should_send else None,
        dedup_key=dedup_key or f"agent-reply:{run_id}:{source_message_id}",
        trace_summary=["finalize_reply:send_command" if should_send else "finalize_reply:no_send"],
        reason="finalized_reply" if should_send else "policy_no_reply",
    )


@dataclass
class FakeReliabilityClient:
    identity: BotIdentity = field(
        default_factory=lambda: BotIdentity(
            user_id=1001,
            username="agent_bot",
            nickname="Agent Bot",
            session_id=5001,
        )
    )
    should_fail_send: bool = False
    delivery_acks: list[int] = field(default_factory=list)
    read_acks: list[tuple[int, int]] = field(default_factory=list)
    sent_private_messages: list[tuple[int, str, str | None]] = field(default_factory=list)
    history_requests: list[tuple[int, int, int, int]] = field(default_factory=list)

    async def pull_offline_messages(self, limit: int = 100) -> list[object]:
        return []

    async def ack_offline_messages(self, message_ids: list[int]) -> None:
        return None

    async def pull_history_messages(
        self,
        *,
        conversation_type: int,
        conversation_id: int,
        before_message_id: int = 0,
        limit: int = 8,
    ) -> list[object]:
        self.history_requests.append(
            (conversation_type, conversation_id, before_message_id, limit)
        )
        return []

    async def send_delivery_ack(self, message_id: int) -> None:
        self.delivery_acks.append(message_id)

    async def send_read_ack(self, conversation_id: int, message_id: int) -> None:
        self.read_acks.append((conversation_id, message_id))

    async def send_private_message(
        self,
        receiver_id: int,
        text: str,
        client_message_id: str | None = None,
    ) -> object:
        if self.should_fail_send:
            raise RuntimeError("LiteIM connection is not available")
        self.sent_private_messages.append((receiver_id, text, client_message_id))
        return object()


class StaticAgentClient:
    def __init__(self, command: AgentReplyCommand) -> None:
        self.command = command
        self.calls = 0

    async def chat_for_message(
        self,
        message: object,
        recent_context: list[object] | tuple[object, ...] = (),
    ) -> AgentReplyCommand:
        self.calls += 1
        return self.command


async def test_botclient_executes_agent_reply_command_and_records_sent(
    tmp_path: Path,
) -> None:
    from bot_client.agent.api import AgentServiceMessageProcessor
    from bot_client.messages.handler import BotMessageHandler
    from bot_client.messages.state import JsonMessageState

    command = _reply_command()
    state = JsonMessageState(tmp_path / "state.json")
    client = FakeReliabilityClient()
    agent = StaticAgentClient(command)
    handler = BotMessageHandler(
        client=client,
        state=state,
        processor=AgentServiceMessageProcessor(agent),
    )

    await handler.handle_packet(
        _packet(MessageType.PrivateMessagePush, _private_message_body())
    )

    assert agent.calls == 1
    assert client.delivery_acks == [7401]
    assert client.read_acks == [(10011002, 7401)]
    assert client.sent_private_messages == [(1002, "agent reply", "pa-run-agent-send")]
    assert state.agent_reply_events[0].status == "sent"
    assert state.agent_reply_events[0].dedup_key == "agent-reply:run-agent-send:7401"
    assert state.agent_reply_events[0].client_message_id == "pa-run-agent-send"
    assert state.agent_reply_events[0].trace_summary == ["finalize_reply:send_command"]


async def test_agent_processor_requests_recent_history_before_calling_agent(
    tmp_path: Path,
) -> None:
    from bot_client.agent.api import AgentServiceMessageProcessor
    from bot_client.messages.handler import BotMessageHandler
    from bot_client.messages.state import JsonMessageState

    class CapturingAgentClient(StaticAgentClient):
        def __init__(self) -> None:
            super().__init__(_reply_command(text="context aware reply"))
            self.recent_context_lengths: list[int] = []

        async def chat_for_message(
            self,
            message: object,
            recent_context: list[object] | tuple[object, ...] = (),
        ) -> AgentReplyCommand:
            self.recent_context_lengths.append(len(recent_context))
            return await super().chat_for_message(message, recent_context)

    client = FakeReliabilityClient()
    agent = CapturingAgentClient()
    handler = BotMessageHandler(
        client=client,
        state=JsonMessageState(tmp_path / "state.json"),
        processor=AgentServiceMessageProcessor(
            agent,
            history_loader=client.pull_history_messages,
            recent_context_limit=8,
        ),
    )

    await handler.handle_packet(
        _packet(MessageType.PrivateMessagePush, _private_message_body(message_id=7409))
    )

    assert client.history_requests == [(1, 10011002, 7409, 8)]
    assert agent.recent_context_lengths == [0]


async def test_retry_after_sent_dedup_key_does_not_send_duplicate(
    tmp_path: Path,
) -> None:
    from bot_client.agent.api import AgentServiceMessageProcessor
    from bot_client.messages.handler import BotMessageHandler
    from bot_client.messages.state import AgentReplyTraceEvent, JsonMessageState

    command = _reply_command(run_id="run-already-sent")
    state = JsonMessageState(tmp_path / "state.json")
    state.record_agent_reply_event(
        AgentReplyTraceEvent(
            status="sent",
            dedup_key=command.dedup_key or "",
            source_message_id=7401,
            receiver_id=1002,
            client_message_id="pa-run-already-sent",
            reason="sent",
        )
    )
    client = FakeReliabilityClient()
    agent = StaticAgentClient(command)
    handler = BotMessageHandler(
        client=client,
        state=state,
        processor=AgentServiceMessageProcessor(agent),
    )

    await handler.handle_packet(
        _packet(MessageType.PrivateMessagePush, _private_message_body())
    )

    assert agent.calls == 1
    assert client.sent_private_messages == []
    assert state.has_processed(7401)
    assert len(state.agent_reply_events) == 1


async def test_agent_noop_command_does_not_send_liteim_message(tmp_path: Path) -> None:
    from bot_client.agent.api import AgentServiceMessageProcessor
    from bot_client.messages.handler import BotMessageHandler
    from bot_client.messages.state import JsonMessageState

    command = _reply_command(should_send=False)
    state = JsonMessageState(tmp_path / "state.json")
    client = FakeReliabilityClient()
    handler = BotMessageHandler(
        client=client,
        state=state,
        processor=AgentServiceMessageProcessor(StaticAgentClient(command)),
    )

    await handler.handle_packet(
        _packet(MessageType.PrivateMessagePush, _private_message_body())
    )

    assert client.delivery_acks == [7401]
    assert client.read_acks == [(10011002, 7401)]
    assert client.sent_private_messages == []
    assert state.agent_reply_events == []
    assert state.has_processed(7401)


async def test_liteim_send_failure_records_failed_trace_and_marks_processed(
    tmp_path: Path,
) -> None:
    from bot_client.agent.api import AgentServiceMessageProcessor
    from bot_client.messages.handler import BotMessageHandler
    from bot_client.messages.state import JsonMessageState

    command = _reply_command(run_id="run-send-failed")
    state = JsonMessageState(tmp_path / "state.json")
    client = FakeReliabilityClient(should_fail_send=True)
    handler = BotMessageHandler(
        client=client,
        state=state,
        processor=AgentServiceMessageProcessor(StaticAgentClient(command)),
    )

    await handler.handle_packet(
        _packet(MessageType.PrivateMessagePush, _private_message_body())
    )

    assert client.sent_private_messages == []
    assert state.has_processed(7401)
    assert state.agent_reply_events[0].status == "failed"
    assert state.agent_reply_events[0].dedup_key == "agent-reply:run-send-failed:7401"
    assert state.agent_reply_events[0].reason == "LiteIM connection is not available"


async def test_self_push_does_not_call_agent_service(tmp_path: Path) -> None:
    from bot_client.agent.api import AgentServiceMessageProcessor
    from bot_client.messages.handler import BotMessageHandler
    from bot_client.messages.state import JsonMessageState

    command = _reply_command()
    client = FakeReliabilityClient()
    agent = StaticAgentClient(command)
    handler = BotMessageHandler(
        client=client,
        state=JsonMessageState(tmp_path / "state.json"),
        processor=AgentServiceMessageProcessor(agent),
    )

    await handler.handle_packet(
        _packet(
            MessageType.PrivateMessagePush,
            _private_message_body(message_id=7402, sender_id=1001, receiver_id=1002),
        )
    )

    assert agent.calls == 0
    assert client.sent_private_messages == []
