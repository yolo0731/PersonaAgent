from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ConsentRecord(BaseModel):
    consent_id: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    subject_user_id: int = Field(ge=1)
    source: str = Field(min_length=1)
    allowed_usage: list[str] = Field(min_length=1)
    forbidden_usage: list[str] = Field(default_factory=list)
    revoked: bool = False
    created_at: str = Field(min_length=1)
    revoked_at: str | None = None

    @model_validator(mode="after")
    def validate_revocation_time(self) -> ConsentRecord:
        if self.revoked and not self.revoked_at:
            raise ValueError("revoked_at is required when revoked is true")
        return self

    @property
    def active(self) -> bool:
        return not self.revoked

    def usage_forbidden(self, usage: str) -> bool:
        return usage in set(self.forbidden_usage)

    def usage_allowed(self, usage: str) -> bool:
        return usage in set(self.allowed_usage)


class ConsentManifest(BaseModel):
    version: int = Field(ge=1)
    records: list[ConsentRecord]

    @classmethod
    def load(cls, path: str | Path) -> ConsentManifest:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def get(self, consent_id: str) -> ConsentRecord | None:
        for record in self.records:
            if record.consent_id == consent_id:
                return record
        return None
