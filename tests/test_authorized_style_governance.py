from __future__ import annotations

import json
from pathlib import Path


def _consent_manifest(*, revoked: bool = False, forbidden: bool = False) -> dict[str, object]:
    return {
        "version": 1,
        "records": [
            {
                "consent_id": "consent-alice-style",
                "persona_id": "alice",
                "subject_user_id": 1002,
                "source": "fixture",
                "allowed_usage": ["style_simulation"],
                "forbidden_usage": ["style_simulation"] if forbidden else [],
                "revoked": revoked,
                "created_at": "2026-05-25T00:00:00Z",
                "revoked_at": "2026-05-25T01:00:00Z" if revoked else None,
            }
        ],
    }


def _raw_record() -> dict[str, object]:
    return {
        "record_id": "raw-1",
        "consent_id": "consent-alice-style",
        "persona_id": "alice",
        "speaker_user_id": 1002,
        "source": "fixture/chat.jsonl",
        "text": "我的邮箱是 alice@example.com，电话是 13800138000。",
        "timestamp_ms": 1_700_000_001_000,
    }


def test_import_rejects_records_without_consent() -> None:
    from agent_service.governance.consent import ConsentManifest
    from agent_service.governance.data_manifest import RawStyleRecord, StyleDataImporter
    from agent_service.governance.pii_redactor import PiiRedactor

    importer = StyleDataImporter(
        manifest=ConsentManifest(version=1, records=[]),
        redactor=PiiRedactor(),
    )

    result = importer.import_records([RawStyleRecord.model_validate(_raw_record())])

    assert result.samples == []
    assert result.report.imported_count == 0
    assert result.report.rejected_count == 1
    assert result.report.rejections[0].reason == "missing_consent"


def test_forbidden_usage_rejects_style_simulation() -> None:
    from agent_service.governance.consent import ConsentManifest
    from agent_service.governance.data_manifest import RawStyleRecord, StyleDataImporter
    from agent_service.governance.pii_redactor import PiiRedactor

    importer = StyleDataImporter(
        manifest=ConsentManifest.model_validate(_consent_manifest(forbidden=True)),
        redactor=PiiRedactor(),
    )

    result = importer.import_records(
        [RawStyleRecord.model_validate(_raw_record())],
        target_usage="style_simulation",
    )

    assert result.samples == []
    assert result.report.rejected_count == 1
    assert result.report.rejections[0].reason == "forbidden_usage"


def test_pii_is_redacted_before_sample_import() -> None:
    from agent_service.governance.consent import ConsentManifest
    from agent_service.governance.data_manifest import RawStyleRecord, StyleDataImporter
    from agent_service.governance.pii_redactor import PiiRedactor

    importer = StyleDataImporter(
        manifest=ConsentManifest.model_validate(_consent_manifest()),
        redactor=PiiRedactor(),
    )

    result = importer.import_records([RawStyleRecord.model_validate(_raw_record())])

    assert result.report.imported_count == 1
    assert result.report.redacted_count == 1
    assert result.samples[0].text == "我的邮箱是 [REDACTED_EMAIL]，电话是 [REDACTED_PHONE]。"
    assert "alice@example.com" not in result.samples[0].text
    assert "13800138000" not in result.samples[0].text
    assert result.samples[0].active is True
    assert result.samples[0].consent_id == "consent-alice-style"


def test_revoked_consent_imports_inactive_sample_for_audit() -> None:
    from agent_service.governance.consent import ConsentManifest
    from agent_service.governance.data_manifest import RawStyleRecord, StyleDataImporter
    from agent_service.governance.pii_redactor import PiiRedactor

    importer = StyleDataImporter(
        manifest=ConsentManifest.model_validate(_consent_manifest(revoked=True)),
        redactor=PiiRedactor(),
    )

    result = importer.import_records([RawStyleRecord.model_validate(_raw_record())])

    assert result.report.imported_count == 1
    assert result.report.inactive_count == 1
    assert result.samples[0].active is False
    assert result.samples[0].revoked is True


def test_import_writes_processed_samples_and_report(tmp_path: Path) -> None:
    from agent_service.governance.consent import ConsentManifest
    from agent_service.governance.data_manifest import RawStyleRecord, StyleDataImporter
    from agent_service.governance.pii_redactor import PiiRedactor

    importer = StyleDataImporter(
        manifest=ConsentManifest.model_validate(_consent_manifest()),
        redactor=PiiRedactor(),
    )
    result = importer.import_records([RawStyleRecord.model_validate(_raw_record())])
    samples_path = tmp_path / "processed_samples.jsonl"
    report_path = tmp_path / "import_report.json"

    result.write(samples_path=samples_path, report_path=report_path)

    sample_lines = samples_path.read_text(encoding="utf-8").splitlines()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(sample_lines) == 1
    assert json.loads(sample_lines[0])["text"] == result.samples[0].text
    assert report["total_records"] == 1
    assert report["imported_count"] == 1
    assert report["redacted_count"] == 1


def test_repository_authorized_style_data_layout_uses_examples_and_ignores_raw() -> None:
    raw_ignore = Path("data/authorized_style_records/raw/.gitignore")
    processed_ignore = Path("data/authorized_style_records/processed/.gitignore")
    consent_example = Path("data/authorized_style_records/consent_manifest.example.json")
    sample_example = Path("data/authorized_style_records/processed/style_samples.example.jsonl")
    report_example = Path("data/authorized_style_records/processed/import_report.example.json")

    assert raw_ignore.read_text(encoding="utf-8").splitlines() == ["*", "!.gitignore"]
    assert "!style_samples.example.jsonl" in processed_ignore.read_text(encoding="utf-8")
    assert json.loads(consent_example.read_text(encoding="utf-8"))["records"][0]["consent_id"]
    assert "[REDACTED_EMAIL]" in sample_example.read_text(encoding="utf-8")
    assert json.loads(report_example.read_text(encoding="utf-8"))["imported_count"] == 1
