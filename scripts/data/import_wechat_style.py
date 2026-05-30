# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_service.governance.consent import ConsentManifest, ConsentRecord
from agent_service.governance.data_manifest import RawStyleRecord, StyleDataImporter
from agent_service.governance.pii_redactor import PiiRedactor
from agent_service.rag.embeddings import MockEmbeddingClient
from agent_service.style.filters import is_learnable_style_text
from agent_service.style.style_store import StyleStore

LOCAL_TZ = timezone(timedelta(hours=8))
DATE_PATTERN = re.compile(
    r"(?P<year>20\d{2})[-/年](?P<month>\d{1,2})[-/月](?P<day>\d{1,2})日?"
    r"(?:[ T](?P<hour>\d{1,2})(?::(?P<minute>\d{1,2}))?)?"
)
SPEAKER_PREFIX_PATTERN = re.compile(
    r"^(?:(?:20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}日?(?:[ T]\d{1,2}:?\d{0,2})?)\s+)?"
    r"(?P<speaker>[^:：\t]{1,32})[:：]\s*(?P<text>.+)$"
)
NON_TEXT_MESSAGES = {
    "[图片]",
    "[表情]",
    "[动画表情]",
    "[视频]",
    "[语音]",
    "[文件]",
    "[位置]",
    "[链接]",
}


@dataclass(frozen=True, slots=True)
class WeChatStyleImportOutput:
    raw_count: int
    imported_count: int
    indexed_count: int
    raw_path: Path
    manifest_path: Path
    samples_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class ManualMessage:
    text: str
    timestamp_ms: int | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert authorized WeChat copied text into PersonaAgent style samples.",
    )
    parser.add_argument("--input-text", required=True, help="UTF-8 text copied from WeChat.")
    parser.add_argument("--speaker-name", help="Only import lines from this speaker when present.")
    parser.add_argument(
        "--persona-id",
        required=True,
        help="Style persona id, for example demo_persona.",
    )
    parser.add_argument(
        "--subject-user-id",
        required=True,
        type=int,
        help="Numeric local subject id.",
    )
    parser.add_argument(
        "--consent-id",
        required=True,
        help="Consent id for these authorized records.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Inclusive start date, e.g. 2000-01-01.",
    )
    parser.add_argument("--end-date", required=True, help="Inclusive end date, e.g. 2000-12-31.")
    parser.add_argument(
        "--source",
        default="wechat_linux_manual_copy",
        help="Source label stored in raw and processed records.",
    )
    parser.add_argument(
        "--raw-jsonl",
        default="data/authorized_style_records/raw/wechat_style.local.jsonl",
    )
    parser.add_argument(
        "--consent-manifest",
        default="data/authorized_style_records/consent_manifest.local.json",
    )
    parser.add_argument(
        "--out-jsonl",
        default="data/authorized_style_records/processed/style_samples.local.jsonl",
    )
    parser.add_argument(
        "--report",
        default="data/authorized_style_records/processed/import_report.local.json",
    )
    parser.add_argument(
        "--index-chroma",
        help="Optional Chroma path to index processed samples for immediate AgentService use.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args(argv)


def build_wechat_style_dataset(args: argparse.Namespace) -> WeChatStyleImportOutput:
    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    if end < start:
        raise ValueError("end-date must be on or after start-date")

    raw_path = Path(args.raw_jsonl)
    manifest_path = Path(args.consent_manifest)
    samples_path = Path(args.out_jsonl)
    report_path = Path(args.report)
    for path in (raw_path, manifest_path, samples_path, report_path):
        _ensure_can_write(path, overwrite=bool(args.overwrite))

    messages = _load_manual_messages(
        Path(args.input_text),
        speaker_name=args.speaker_name,
        start_date=start,
        end_date=end,
    )
    if not messages:
        raise ValueError("no WeChat text messages matched the speaker/date filters")

    consent = ConsentRecord(
        consent_id=args.consent_id,
        persona_id=args.persona_id,
        subject_user_id=args.subject_user_id,
        source=_source_label(args.source, start, end),
        allowed_usage=["style_simulation"],
        forbidden_usage=[
            "hidden_impersonation",
            "real_world_commitments",
            "privacy_disclosure",
        ],
        revoked=False,
        created_at=datetime.now(tz=LOCAL_TZ).isoformat(),
        revoked_at=None,
    )
    manifest = ConsentManifest(version=1, records=[consent])
    raw_records = _raw_records_from_messages(messages, args=args, start_date=start, end_date=end)

    _write_json(raw_path, [record.model_dump(mode="json") for record in raw_records], jsonl=True)
    _write_json(manifest_path, manifest.model_dump(mode="json"), jsonl=False)

    result = StyleDataImporter(manifest=manifest, redactor=PiiRedactor()).import_records(
        raw_records,
        target_usage="style_simulation",
    )
    result.write(samples_path=samples_path, report_path=report_path)

    indexed_count = 0
    if args.index_chroma:
        indexed_count = StyleStore(
            chroma_path=args.index_chroma,
            embedding_client=MockEmbeddingClient(),
        ).replace_samples(result.samples)

    return WeChatStyleImportOutput(
        raw_count=len(raw_records),
        imported_count=result.report.imported_count,
        indexed_count=indexed_count,
        raw_path=raw_path,
        manifest_path=manifest_path,
        samples_path=samples_path,
        report_path=report_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = build_wechat_style_dataset(args)
    except Exception as exc:
        print(f"WeChat style import failed: {exc}", file=sys.stderr)
        return 1
    print(
        "WeChat style import complete "
        f"raw={output.raw_count} "
        f"imported={output.imported_count} "
        f"indexed={output.indexed_count} "
        f"samples={output.samples_path}"
    )
    return 0


def _load_manual_messages(
    path: Path,
    *,
    speaker_name: str | None,
    start_date: date,
    end_date: date,
) -> list[ManualMessage]:
    messages: list[ManualMessage] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        message = _parse_line(
            line,
            speaker_name=speaker_name,
            start_date=start_date,
            end_date=end_date,
            fallback_index=index,
        )
        if message is not None:
            messages.append(message)
    return messages


def _parse_line(
    line: str,
    *,
    speaker_name: str | None,
    start_date: date,
    end_date: date,
    fallback_index: int,
) -> ManualMessage | None:
    stripped = line.strip()
    if not stripped or stripped in NON_TEXT_MESSAGES:
        return None

    parsed_at = _timestamp_from_text(stripped)
    if parsed_at is not None and not _date_in_range(parsed_at, start_date, end_date):
        return None

    text = _extract_message_text(stripped, speaker_name=speaker_name)
    if text is None or text in NON_TEXT_MESSAGES or not is_learnable_style_text(text):
        return None

    timestamp_ms = parsed_at
    if timestamp_ms is None:
        timestamp_ms = _date_to_ms(start_date, fallback_index)
    return ManualMessage(text=text, timestamp_ms=timestamp_ms)


def _extract_message_text(line: str, *, speaker_name: str | None) -> str | None:
    tab_parts = [part.strip() for part in line.split("\t")]
    if len(tab_parts) >= 3:
        speaker = tab_parts[1]
        if speaker_name is not None and speaker != speaker_name:
            return None
        return "\t".join(tab_parts[2:]).strip() or None

    prefix_match = SPEAKER_PREFIX_PATTERN.match(line)
    if prefix_match is not None:
        speaker = prefix_match.group("speaker").strip()
        if speaker_name is not None and speaker != speaker_name:
            return None
        return prefix_match.group("text").strip() or None

    return line


def _raw_records_from_messages(
    messages: Sequence[ManualMessage],
    *,
    args: argparse.Namespace,
    start_date: date,
    end_date: date,
) -> list[RawStyleRecord]:
    source = _source_label(args.source, start_date, end_date)
    records: list[RawStyleRecord] = []
    for index, message in enumerate(messages):
        records.append(
            RawStyleRecord(
                record_id=f"wechat-{args.persona_id}-{index + 1:06d}",
                consent_id=args.consent_id,
                persona_id=args.persona_id,
                speaker_user_id=args.subject_user_id,
                source=source,
                text=message.text,
                timestamp_ms=message.timestamp_ms or _date_to_ms(start_date, index),
            )
        )
    return records


def _timestamp_from_text(text: str) -> int | None:
    match = DATE_PATTERN.search(text)
    if match is None:
        return None
    hour = int(match.group("hour") or 0)
    minute = int(match.group("minute") or 0)
    value = datetime(
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
        hour,
        minute,
        tzinfo=LOCAL_TZ,
    )
    return int(value.timestamp() * 1000)


def _date_in_range(timestamp_ms: int, start_date: date, end_date: date) -> bool:
    value = datetime.fromtimestamp(timestamp_ms / 1000, tz=LOCAL_TZ).date()
    return start_date <= value <= end_date


def _date_to_ms(value: date, offset_seconds: int) -> int:
    stamp = datetime.combine(value, time.min, tzinfo=LOCAL_TZ) + timedelta(
        seconds=offset_seconds
    )
    return int(stamp.timestamp() * 1000)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value.replace("/", "-"))


def _source_label(source: str, start_date: date, end_date: date) -> str:
    return f"{source}:{start_date.isoformat()}..{end_date.isoformat()}"


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: object, *, jsonl: bool) -> None:
    if jsonl:
        assert isinstance(payload, list)
        path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in payload)
            + ("\n" if payload else ""),
            encoding="utf-8",
        )
        return
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
