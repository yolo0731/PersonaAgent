from __future__ import annotations

from dataclasses import dataclass

from bot_client.messages.handler import MessageProcessingResult
from bot_client.protocol.parsers import IncomingMessage


@dataclass(frozen=True, slots=True)
class EchoMessageProcessor:
    enabled: bool = True

    def __call__(self, message: IncomingMessage) -> MessageProcessingResult:
        if not self.enabled:
            return MessageProcessingResult()
        return MessageProcessingResult(reply_text=message.text)
