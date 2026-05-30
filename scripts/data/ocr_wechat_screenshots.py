# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_TZ = timezone(timedelta(hours=8))
DATE_PATTERN = re.compile(
    r"(?P<year>20\d{2})[-/年](?P<month>\d{1,2})[-/月](?P<day>\d{1,2})日?"
    r"(?:\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{1,2}))?)?"
)
SHORT_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<year>\d{2})/(?P<month>\d{1,2})/(?P<day>\d{1,2})(?!\d)"
)
NON_TEXT_LINES = {
    "[图片]",
    "[表情]",
    "[动画表情]",
    "[视频]",
    "[语音]",
    "[文件]",
    "[位置]",
    "[链接]",
}

OcrReader = Callable[[Path], str]
RapidOcrReader = Callable[[Path], list[tuple[list[list[float]], str, float]]]
ToolLookup = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class OcrLine:
    role: str
    speaker: str
    text: str
    source_image: str
    line_no: int
    timestamp_ms: int | None


@dataclass(frozen=True, slots=True)
class OcrResult:
    line_count: int
    output_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR WeChat screenshots into role-separated local JSONL lines.",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--output-jsonl",
        default="data/authorized_style_records/raw/wechat_demo_persona_ocr_lines.local.jsonl",
    )
    parser.add_argument("--self-speaker-name", required=True)
    parser.add_argument("--target-speaker-name", required=True)
    parser.add_argument(
        "--engine",
        choices=("rapidocr", "tesseract"),
        default="rapidocr",
        help="rapidocr uses text boxes and is more reliable for WeChat chat-history pages.",
    )
    parser.add_argument("--tesseract-lang", default="chi_sim+eng")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def ocr_wechat_screenshots(
    args: argparse.Namespace,
    *,
    ocr_reader: OcrReader | None = None,
    rapidocr_reader: RapidOcrReader | None = None,
    tool_lookup: ToolLookup = shutil.which,
) -> OcrResult:
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_jsonl)
    _ensure_can_write(output_path, overwrite=bool(args.overwrite))
    images = sorted(input_dir.glob("*.png"))
    if not images:
        raise ValueError(f"no PNG screenshots found in {input_dir}")

    records: list[OcrLine] = []
    if ocr_reader is not None or args.engine == "tesseract":
        reader = ocr_reader or _build_tesseract_reader(args.tesseract_lang, tool_lookup)
        for image in images:
            records.extend(
                _records_from_text(
                    reader(image),
                    image_name=image.name,
                    self_speaker=args.self_speaker_name,
                    target_speaker=args.target_speaker_name,
                )
            )
    else:
        reader = rapidocr_reader or _build_rapidocr_reader()
        for image in images:
            records.extend(
                _records_from_rapidocr(
                    reader(image),
                    image_path=image,
                    image_name=image.name,
                    self_speaker=args.self_speaker_name,
                    target_speaker=args.target_speaker_name,
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            json.dumps(_record_payload(record), ensure_ascii=False, sort_keys=True)
            for record in records
        )
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    return OcrResult(line_count=len(records), output_path=output_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = ocr_wechat_screenshots(args)
    except Exception as exc:
        print(f"WeChat OCR failed: {exc}", file=sys.stderr)
        return 1
    print(f"WeChat OCR complete lines={result.line_count} output={result.output_path}")
    return 0


def _records_from_text(
    text: str,
    *,
    image_name: str,
    self_speaker: str,
    target_speaker: str,
) -> list[OcrLine]:
    records: list[OcrLine] = []
    current_speaker: str | None = None
    current_timestamp: int | None = None
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed_timestamp = _timestamp_from_line(line)
        if parsed_timestamp is not None:
            current_timestamp = parsed_timestamp
            continue
        if line == self_speaker:
            current_speaker = self_speaker
            continue
        if line == target_speaker:
            current_speaker = target_speaker
            continue
        if current_speaker is None or line in NON_TEXT_LINES:
            continue
        role = "target" if current_speaker == target_speaker else "self"
        records.append(
            OcrLine(
                role=role,
                speaker=current_speaker,
                text=line,
                source_image=image_name,
                line_no=line_no,
                timestamp_ms=current_timestamp,
            )
        )
    return records


def _records_from_rapidocr(
    items: Sequence[tuple[list[list[float]], str, float]],
    *,
    image_path: Path | None = None,
    image_name: str,
    self_speaker: str,
    target_speaker: str,
) -> list[OcrLine]:
    lines = sorted(
        [_box_line(item) for item in items if _box_line(item) is not None],
        key=lambda item: (item.y_min, item.x_min),
    )
    left_lines = [
        line
        for line in lines
        if line.x_min < 700
        and line.y_min > 100
        and not _is_noise_text(line.text)
    ]
    records: list[OcrLine] = []
    current_speaker: str | None = None
    current_timestamp: int | None = None
    last_content_y: float | None = None
    image = _load_rgb_image(image_path)
    for index, line in enumerate(left_lines):
        timestamp = _timestamp_from_line(line.text)
        if timestamp is not None:
            current_timestamp = timestamp
            continue
        if _is_self_label(line.text, self_speaker):
            current_speaker = self_speaker
            last_content_y = line.y_min
            continue
        if _is_target_label(line.text, target_speaker) or _looks_like_speaker_header(
            line,
            _next_left_line(left_lines, index),
        ):
            current_speaker = target_speaker
            last_content_y = line.y_min
            continue
        if _is_inside_embedded_media(line, image):
            continue
        if current_speaker is None:
            current_speaker = target_speaker
        elif (
            current_speaker == self_speaker
            and last_content_y is not None
            and line.y_min - last_content_y > 42
        ):
            current_speaker = target_speaker
        role = "target" if current_speaker == target_speaker else "self"
        records.append(
            OcrLine(
                role=role,
                speaker=current_speaker,
                text=line.text,
                source_image=image_name,
                line_no=index + 1,
                timestamp_ms=current_timestamp,
            )
        )
        last_content_y = line.y_min
    return records


def _timestamp_from_line(line: str) -> int | None:
    match = DATE_PATTERN.search(line)
    if match is not None:
        return _timestamp_from_parts(
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
            hour=int(match.group("hour") or 0),
            minute=int(match.group("minute") or 0),
        )
    short_match = SHORT_DATE_PATTERN.search(line)
    if short_match is None:
        return None
    return _timestamp_from_parts(
        year=2000 + int(short_match.group("year")),
        month=int(short_match.group("month")),
        day=int(short_match.group("day")),
        hour=0,
        minute=0,
    )


def _timestamp_from_parts(
    *,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
) -> int:
    value = datetime(
        year,
        month,
        day,
        hour,
        minute,
        tzinfo=LOCAL_TZ,
    )
    return int(value.timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class _BoxLine:
    text: str
    score: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def _box_line(item: tuple[list[list[float]], str, float]) -> _BoxLine | None:
    box, text, score = item
    cleaned = text.strip()
    if not cleaned or score < 0.45:
        return None
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return _BoxLine(
        text=cleaned,
        score=score,
        x_min=min(xs),
        y_min=min(ys),
        x_max=max(xs),
        y_max=max(ys),
    )


def _next_left_line(lines: Sequence[_BoxLine], index: int) -> _BoxLine | None:
    return lines[index + 1] if index + 1 < len(lines) else None


def _looks_like_speaker_header(line: _BoxLine, next_line: _BoxLine | None) -> bool:
    if next_line is None:
        return False
    if len(line.text) > 6:
        return False
    vertical_gap = next_line.y_min - line.y_min
    same_column = abs(next_line.x_min - line.x_min) < 35
    return 12 <= vertical_gap <= 42 and same_column and len(next_line.text) > 1


def _load_rgb_image(path: Path | None) -> Any | None:
    if path is None:
        return None
    with suppress(ImportError, FileNotFoundError, OSError):
        from PIL import Image

        return Image.open(path).convert("RGB")
    return None


def _is_inside_embedded_media(line: _BoxLine, image: Any | None) -> bool:
    if image is None:
        return False
    left = max(0, int(line.x_min) - 18)
    top = max(0, int(line.y_min) - 18)
    right = min(image.width, int(line.x_max) + 18)
    bottom = min(image.height, int(line.y_max) + 18)
    if right <= left or bottom <= top:
        return False
    crop = image.crop((left, top, right, bottom))
    pixels = list(
        crop.get_flattened_data() if hasattr(crop, "get_flattened_data") else crop.getdata()
    )
    if not pixels:
        return False
    page_gray_pixels = sum(1 for pixel in pixels if _is_wechat_page_gray(pixel))
    dark_pixels = sum(1 for pixel in pixels if max(pixel) < 115)
    saturated_or_colored_pixels = sum(
        1 for pixel in pixels if max(pixel) - min(pixel) > 35 or max(pixel) < 185
    )
    total = len(pixels)
    page_gray_ratio = page_gray_pixels / total
    dark_ratio = dark_pixels / total
    saturated_or_colored_ratio = saturated_or_colored_pixels / total
    return (
        page_gray_ratio < 0.40
        or dark_ratio >= 0.28
        or saturated_or_colored_ratio >= 0.55
    )


def _is_wechat_page_gray(pixel: tuple[int, int, int]) -> bool:
    red, green, blue = pixel
    return (
        220 <= red <= 248
        and 220 <= green <= 248
        and 220 <= blue <= 248
        and max(pixel) - min(pixel) <= 12
    )


def _is_self_label(text: str, self_speaker: str) -> bool:
    compact = _compact_text(text)
    return _compact_text(self_speaker) in compact or compact in {"bae", "be", "当前用户"}


def _is_target_label(text: str, target_speaker: str) -> bool:
    compact = _compact_text(text)
    if _compact_text(target_speaker) in compact:
        return True
    # RapidOCR often confuses the small gray target nickname, but it stays a short header.
    return compact in {"曦意", "曦惠", "随意", "唯息", "唯惠", "睡意", "睡患", "目标样本"}


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _is_noise_text(text: str) -> bool:
    compact = _compact_text(text)
    return (
        compact in {"q搜索", "搜索", "定位到聊天位置", "定位到著天位置"}
        or compact in {item.strip("[]").casefold() for item in NON_TEXT_LINES}
    )


def _looks_like_date(text: str) -> bool:
    compact = _compact_text(text)
    return bool(
        re.fullmatch(r"(?:星期[一二三四五六日天]|昨天|\d{2}/\d{1,2}/\d{1,2})", compact)
    )


def _build_rapidocr_reader() -> RapidOcrReader:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "rapidocr-onnxruntime is required for --engine rapidocr; "
            "install it in the agent conda environment"
        ) from exc

    engine = RapidOCR()

    def read(image: Path) -> list[tuple[list[list[float]], str, float]]:
        result, _elapsed = engine(str(image))
        return list(result or [])

    return read


def _build_tesseract_reader(lang: str, tool_lookup: ToolLookup) -> OcrReader:
    if tool_lookup("tesseract") is None:
        raise RuntimeError("required tool not found in PATH: tesseract")

    def read(image: Path) -> str:
        completed = subprocess.run(
            ["tesseract", str(image), "stdout", "-l", lang],
            check=True,
            text=True,
            capture_output=True,
        )
        return completed.stdout

    return read


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)


def _record_payload(record: OcrLine) -> dict[str, object]:
    payload: dict[str, object] = {
        "role": record.role,
        "speaker": record.speaker,
        "text": record.text,
        "source_image": record.source_image,
        "line_no": record.line_no,
    }
    if record.timestamp_ms is not None:
        payload["timestamp_ms"] = record.timestamp_ms
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
