# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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


@dataclass(frozen=True, slots=True)
class OcrMessage:
    sequence_index: int
    role: str
    speaker: str
    text: str
    source_image: str
    line_no: int
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class StyleFromOcrOutput:
    self_count: int
    target_count: int
    pair_count: int
    indexed_count: int
    samples_path: Path
    pairs_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build authorized target-speaker style samples from OCR JSONL.",
    )
    parser.add_argument("--ocr-jsonl", required=True)
    parser.add_argument("--self-speaker-name", required=True)
    parser.add_argument("--target-speaker-name", required=True)
    parser.add_argument("--persona-id", required=True)
    parser.add_argument("--subject-user-id", required=True, type=int)
    parser.add_argument("--consent-id", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--raw-jsonl",
        default="data/authorized_style_records/raw/wechat_demo_persona_ocr.local.jsonl",
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
        "--pairs-jsonl",
        default="data/authorized_style_records/processed/style_pairs.local.jsonl",
    )
    parser.add_argument(
        "--report",
        default="data/authorized_style_records/processed/import_report.local.json",
    )
    parser.add_argument("--index-chroma", help="Optional Chroma path for immediate indexing.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build_style_from_ocr(args: argparse.Namespace) -> StyleFromOcrOutput:
    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    if end < start:
        raise ValueError("end-date must be on or after start-date")

    raw_path = Path(args.raw_jsonl)
    manifest_path = Path(args.consent_manifest)
    samples_path = Path(args.out_jsonl)
    pairs_path = Path(args.pairs_jsonl)
    report_path = Path(args.report)
    for path in (raw_path, manifest_path, samples_path, pairs_path, report_path):
        _ensure_can_write(path, overwrite=bool(args.overwrite))

    messages = _dedupe_messages(
        _load_ocr_messages(Path(args.ocr_jsonl), start_date=start, end_date=end)
    )
    self_messages = [
        message
        for message in messages
        if message.role == "self" and message.speaker == args.self_speaker_name
    ]
    target_messages = [
        message
        for message in messages
        if message.role == "target" and message.speaker == args.target_speaker_name
    ]
    if not target_messages:
        raise ValueError("no target speaker messages matched OCR/date filters")

    consent = ConsentRecord(
        consent_id=args.consent_id,
        persona_id=args.persona_id,
        subject_user_id=args.subject_user_id,
        source=_source_label("wechat_ocr", start, end),
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
    raw_records = [
        RawStyleRecord(
            record_id=f"wechat-{args.persona_id}-{index + 1:06d}",
            consent_id=args.consent_id,
            persona_id=args.persona_id,
            speaker_user_id=args.subject_user_id,
            source=_source_label("wechat_ocr", start, end),
            text=message.text,
            timestamp_ms=message.timestamp_ms,
        )
        for index, message in enumerate(target_messages)
    ]
    pairs = _pair_self_to_target(
        messages,
        persona_id=args.persona_id,
        self_speaker=args.self_speaker_name,
        target_speaker=args.target_speaker_name,
    )

    _write_jsonl(raw_path, [record.model_dump(mode="json") for record in raw_records])
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    _write_jsonl(pairs_path, pairs)

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

    return StyleFromOcrOutput(
        self_count=len(self_messages),
        target_count=len(target_messages),
        pair_count=len(pairs),
        indexed_count=indexed_count,
        samples_path=samples_path,
        pairs_path=pairs_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = build_style_from_ocr(args)
    except Exception as exc:
        print(f"WeChat OCR style build failed: {exc}", file=sys.stderr)
        return 1
    print(
        "WeChat OCR style build complete "
        f"self={output.self_count} "
        f"target={output.target_count} "
        f"pairs={output.pair_count} "
        f"indexed={output.indexed_count} "
        f"samples={output.samples_path}"
    )
    return 0


def _load_ocr_messages(path: Path, *, start_date: date, end_date: date) -> list[OcrMessage]:
    messages: list[OcrMessage] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        text = str(raw["text"]).strip()
        if not is_learnable_style_text(text):
            continue
        timestamp_ms = int(raw.get("timestamp_ms") or _date_to_ms(start_date, line_no))
        if not _date_in_range(timestamp_ms, start_date, end_date):
            continue
        messages.append(
            OcrMessage(
                sequence_index=len(messages),
                role=str(raw["role"]),
                speaker=str(raw["speaker"]),
                text=text,
                source_image=str(raw.get("source_image", "")),
                line_no=int(raw.get("line_no", line_no)),
                timestamp_ms=timestamp_ms,
            )
        )
    return messages


def _dedupe_messages(messages: Sequence[OcrMessage]) -> list[OcrMessage]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[OcrMessage] = []
    for message in messages:
        key = (message.role, message.speaker, message.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(message)
    return deduped


def _pair_self_to_target(
    messages: Sequence[OcrMessage],
    *,
    persona_id: str,
    self_speaker: str,
    target_speaker: str,
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    last_self: OcrMessage | None = None
    for message in sorted(messages, key=lambda item: item.sequence_index):
        if message.role == "self" and message.speaker == self_speaker:
            last_self = message
            continue
        if message.role != "target" or message.speaker != target_speaker:
            continue
        if last_self is None:
            continue
        target = message
        source = last_self
        pairs.append(
            {
                "pair_id": f"wechat-{persona_id}-pair-{len(pairs) + 1:06d}",
                "persona_id": persona_id,
                "self_speaker": self_speaker,
                "target_speaker": target_speaker,
                "self_text": source.text,
                "target_reply": target.text,
                "timestamp_ms": target.timestamp_ms,
                "source_image": target.source_image,
            }
        )
    return pairs


def _date_in_range(timestamp_ms: int, start_date: date, end_date: date) -> bool:
    value = datetime.fromtimestamp(timestamp_ms / 1000, tz=LOCAL_TZ).date()
    return start_date <= value <= end_date


def _date_to_ms(value: date, offset_seconds: int) -> int:
    stamp = datetime.combine(value, datetime.min.time(), tzinfo=LOCAL_TZ) + timedelta(
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
