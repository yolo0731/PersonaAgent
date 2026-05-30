from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agent_service.governance.data_manifest import ProcessedStyleSample
from agent_service.governance.pii_redactor import PiiRedactor
from agent_service.style.filters import is_learnable_style_text
from agent_service.style.style_store import StyleStore


class StyleLearningStore:
    """Append safe runtime replies into the local authorized style dataset."""

    def __init__(
        self,
        *,
        samples_path: str | Path,
        style_store: StyleStore,
        persona_id: str,
        consent_id: str,
        subject_user_id: int,
        redactor: PiiRedactor | None = None,
    ) -> None:
        self._samples_path = Path(samples_path)
        self._style_store = style_store
        self._persona_id = persona_id
        self._consent_id = consent_id
        self._subject_user_id = subject_user_id
        self._redactor = redactor or PiiRedactor()

    def learn_reply(
        self,
        *,
        text: str,
        source_message_id: int,
        timestamp_ms: int,
    ) -> ProcessedStyleSample | None:
        cleaned = text.strip()
        if not is_learnable_style_text(cleaned):
            return None

        sample_id = self._sample_id(source_message_id)
        if sample_id in self._existing_sample_ids():
            return None

        redaction = self._redactor.redact(cleaned)
        sample = ProcessedStyleSample(
            sample_id=sample_id,
            record_id=f"runtime-reply-{source_message_id}",
            consent_id=self._consent_id,
            persona_id=self._persona_id,
            speaker_user_id=self._subject_user_id,
            source=f"agent_runtime_reply:{source_message_id}",
            text=redaction.text,
            allowed_usage=["style_feedback"],
            forbidden_usage=[
                "style_retrieval_without_review",
                "hidden_impersonation",
                "real_world_commitments",
                "privacy_disclosure",
            ],
            active=True,
            revoked=False,
            pii_redactions=redaction.replacements,
            timestamp_ms=timestamp_ms or _now_ms(),
        )
        self._append_sample(sample)
        return sample

    def _sample_id(self, source_message_id: int) -> str:
        return f"runtime-style-{self._consent_id}-{source_message_id}"

    def _existing_sample_ids(self) -> set[str]:
        if not self._samples_path.exists():
            return set()
        existing: set[str] = set()
        for line in self._samples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = raw.get("sample_id") if isinstance(raw, dict) else None
            if isinstance(sample_id, str):
                existing.add(sample_id)
        return existing

    def _append_sample(self, sample: ProcessedStyleSample) -> None:
        self._samples_path.parent.mkdir(parents=True, exist_ok=True)
        with self._samples_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    sample.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)
