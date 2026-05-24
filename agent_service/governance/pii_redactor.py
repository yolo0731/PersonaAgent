from __future__ import annotations

import re

from pydantic import BaseModel

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


class RedactionResult(BaseModel):
    text: str
    replacements: dict[str, int]

    @property
    def redacted(self) -> bool:
        return any(count > 0 for count in self.replacements.values())


class PiiRedactor:
    def redact(self, text: str) -> RedactionResult:
        redacted, email_count = EMAIL_PATTERN.subn("[REDACTED_EMAIL]", text)
        redacted, phone_count = PHONE_PATTERN.subn("[REDACTED_PHONE]", redacted)
        redacted, id_count = ID_CARD_PATTERN.subn("[REDACTED_ID]", redacted)
        return RedactionResult(
            text=redacted,
            replacements={
                "email": email_count,
                "phone": phone_count,
                "id_card": id_count,
            },
        )
