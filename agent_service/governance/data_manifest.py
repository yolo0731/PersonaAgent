from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from agent_service.governance.consent import ConsentManifest, ConsentRecord
from agent_service.governance.pii_redactor import PiiRedactor


class RawStyleRecord(BaseModel):
    record_id: str = Field(min_length=1)
    consent_id: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    speaker_user_id: int = Field(ge=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)


class ProcessedStyleSample(BaseModel):
    sample_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    consent_id: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    speaker_user_id: int = Field(ge=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    allowed_usage: list[str]
    forbidden_usage: list[str]
    active: bool
    revoked: bool
    pii_redactions: dict[str, int]
    timestamp_ms: int = Field(ge=0)


class ImportRejection(BaseModel):
    record_id: str = Field(min_length=1)
    consent_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class DataImportReport(BaseModel):
    total_records: int = Field(ge=0)
    imported_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    redacted_count: int = Field(ge=0)
    inactive_count: int = Field(ge=0)
    rejections: list[ImportRejection] = Field(default_factory=list)


class StyleImportResult(BaseModel):
    samples: list[ProcessedStyleSample]
    report: DataImportReport

    def write(self, *, samples_path: str | Path, report_path: str | Path) -> None:
        sample_file = Path(samples_path)
        report_file = Path(report_path)
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text(
            "\n".join(
                json.dumps(sample.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                for sample in self.samples
            )
            + ("\n" if self.samples else ""),
            encoding="utf-8",
        )
        report_file.write_text(
            json.dumps(
                self.report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


class StyleDataImporter:
    def __init__(self, *, manifest: ConsentManifest, redactor: PiiRedactor) -> None:
        self._manifest = manifest
        self._redactor = redactor

    def import_records(
        self,
        records: list[RawStyleRecord],
        *,
        target_usage: str = "style_simulation",
    ) -> StyleImportResult:
        samples: list[ProcessedStyleSample] = []
        rejections: list[ImportRejection] = []
        redacted_count = 0
        inactive_count = 0

        for index, record in enumerate(records):
            consent = self._manifest.get(record.consent_id)
            rejection_reason = _rejection_reason(consent, target_usage)
            if rejection_reason is not None:
                rejections.append(
                    ImportRejection(
                        record_id=record.record_id,
                        consent_id=record.consent_id,
                        reason=rejection_reason,
                    )
                )
                continue

            assert consent is not None
            redaction = self._redactor.redact(record.text)
            redacted_count += int(redaction.redacted)
            active = consent.active
            inactive_count += int(not active)
            samples.append(
                ProcessedStyleSample(
                    sample_id=f"style-{record.consent_id}-{index:04d}",
                    record_id=record.record_id,
                    consent_id=record.consent_id,
                    persona_id=record.persona_id,
                    speaker_user_id=record.speaker_user_id,
                    source=record.source,
                    text=redaction.text,
                    allowed_usage=list(consent.allowed_usage),
                    forbidden_usage=list(consent.forbidden_usage),
                    active=active,
                    revoked=consent.revoked,
                    pii_redactions=redaction.replacements,
                    timestamp_ms=record.timestamp_ms,
                )
            )

        report = DataImportReport(
            total_records=len(records),
            imported_count=len(samples),
            rejected_count=len(rejections),
            redacted_count=redacted_count,
            inactive_count=inactive_count,
            rejections=rejections,
        )
        return StyleImportResult(samples=samples, report=report)


def _rejection_reason(consent: ConsentRecord | None, target_usage: str) -> str | None:
    if consent is None:
        return "missing_consent"
    if consent.usage_forbidden(target_usage):
        return "forbidden_usage"
    if not consent.usage_allowed(target_usage):
        return "usage_not_allowed"
    return None
