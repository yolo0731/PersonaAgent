from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bot_client.protocol_parser import ReceiptTraceEvent


class JsonMessageState:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._processed_message_ids: set[int] = set()
        self._receipts: list[ReceiptTraceEvent] = []
        self._load()

    @property
    def receipts(self) -> list[ReceiptTraceEvent]:
        return list(self._receipts)

    def has_processed(self, message_id: int) -> bool:
        return message_id in self._processed_message_ids

    def mark_processed(self, message_id: int) -> None:
        self._processed_message_ids.add(message_id)
        self._save()

    def record_receipt(self, event: ReceiptTraceEvent) -> None:
        self._receipts.append(event)
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
        }
        tmp_path = self._path.with_name(f"{self._path.name}.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self._path)
