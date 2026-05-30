from __future__ import annotations

import json
from pathlib import Path


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_capture_wechat_chat_activates_window_screenshots_and_scrolls(
    tmp_path: Path,
) -> None:
    from scripts.data.capture_wechat_chat import capture_chat_screenshots, parse_args

    commands: list[list[str]] = []

    def fake_runner(command: list[str]):
        commands.append(command)
        if command[:2] == ["gnome-screenshot", "-f"]:
            Path(command[2]).write_bytes(f"image-{len(commands)}".encode())
        return ""

    args = parse_args(
        [
            "--window-id",
            "12345",
            "--output-dir",
            str(tmp_path / "screens"),
            "--max-shots",
            "2",
            "--interval-seconds",
            "0",
            "--scroll-clicks",
            "3",
            "--scroll-x",
            "400",
            "--scroll-y",
            "300",
            "--screenshot-tool",
            "gnome-screenshot",
            "--overwrite",
        ]
    )

    result = capture_chat_screenshots(
        args,
        command_runner=fake_runner,
        tool_lookup=lambda name: f"/usr/bin/{name}",
        sleeper=lambda _seconds: None,
    )

    assert [path.name for path in result.images] == ["shot_000001.png", "shot_000002.png"]
    assert commands == [
        ["xdotool", "windowactivate", "--sync", "12345"],
        ["xdotool", "mousemove", "--window", "12345", "400", "300"],
        ["gnome-screenshot", "-f", str(tmp_path / "screens" / "shot_000001.png")],
        ["xdotool", "click", "--repeat", "3", "4"],
        ["gnome-screenshot", "-f", str(tmp_path / "screens" / "shot_000002.png")],
        ["xdotool", "click", "--repeat", "3", "4"],
    ]


def test_capture_wechat_chat_appends_after_existing_screenshots(
    tmp_path: Path,
) -> None:
    from scripts.data.capture_wechat_chat import capture_chat_screenshots, parse_args

    output_dir = tmp_path / "screens"
    output_dir.mkdir()
    (output_dir / "shot_000001.png").write_bytes(b"existing")
    commands: list[list[str]] = []

    def fake_runner(command: list[str]):
        commands.append(command)
        if command[:2] == ["gnome-screenshot", "-f"]:
            Path(command[2]).write_bytes(b"new")
        return ""

    args = parse_args(
        [
            "--window-id",
            "12345",
            "--output-dir",
            str(output_dir),
            "--max-shots",
            "2",
            "--interval-seconds",
            "0",
            "--scroll-clicks",
            "3",
            "--screenshot-tool",
            "gnome-screenshot",
            "--append",
        ]
    )

    result = capture_chat_screenshots(
        args,
        command_runner=fake_runner,
        tool_lookup=lambda name: f"/usr/bin/{name}",
        sleeper=lambda _seconds: None,
    )

    assert [path.name for path in result.images] == ["shot_000002.png", "shot_000003.png"]
    assert ["gnome-screenshot", "-f", str(output_dir / "shot_000002.png")] in commands
    assert ["gnome-screenshot", "-f", str(output_dir / "shot_000003.png")] in commands


def test_ocr_wechat_screenshots_marks_self_and_target_roles(tmp_path: Path) -> None:
    from scripts.data.ocr_wechat_screenshots import ocr_wechat_screenshots, parse_args

    image_dir = tmp_path / "screens"
    image_dir.mkdir()
    (image_dir / "shot_000001.png").write_bytes(b"fake")
    output_jsonl = tmp_path / "ocr.jsonl"

    args = parse_args(
        [
            "--input-dir",
            str(image_dir),
            "--output-jsonl",
            str(output_jsonl),
            "--self-speaker-name",
            "当前用户",
            "--target-speaker-name",
            "目标样本",
            "--overwrite",
        ]
    )

    result = ocr_wechat_screenshots(
        args,
        ocr_reader=lambda _image: "\n".join(
            [
                "2025年5月18日 14:57",
                "当前用户",
                "示例上下文",
                "目标样本",
                "示例回复一",
                "[图片]",
            ]
        ),
        tool_lookup=lambda name: f"/usr/bin/{name}",
    )

    records = _jsonl(output_jsonl)
    assert result.line_count == 2
    assert records == [
        {
            "role": "self",
            "speaker": "当前用户",
            "text": "示例上下文",
            "source_image": "shot_000001.png",
            "line_no": 3,
            "timestamp_ms": 1747551420000,
        },
        {
            "role": "target",
            "speaker": "目标样本",
            "text": "示例回复一",
            "source_image": "shot_000001.png",
            "line_no": 5,
            "timestamp_ms": 1747551420000,
        },
    ]


def test_ocr_wechat_screenshots_rapidocr_recovers_missing_target_header(
    tmp_path: Path,
) -> None:
    from scripts.data.ocr_wechat_screenshots import ocr_wechat_screenshots, parse_args

    image_dir = tmp_path / "screens"
    image_dir.mkdir()
    (image_dir / "shot_000001.png").write_bytes(b"fake")
    output_jsonl = tmp_path / "ocr.jsonl"

    args = parse_args(
        [
            "--input-dir",
            str(image_dir),
            "--output-jsonl",
            str(output_jsonl),
            "--self-speaker-name",
            "当前用户",
            "--target-speaker-name",
            "目标样本",
            "--engine",
            "rapidocr",
            "--overwrite",
        ]
    )

    def box(text: str, x: float, y: float) -> tuple[list[list[float]], str, float]:
        return ([[x, y], [x + 120, y], [x + 120, y + 24], [x, y + 24]], text, 0.95)

    result = ocr_wechat_screenshots(
        args,
        rapidocr_reader=lambda _image: [
            box("当前用户", 65, 120),
            box("25/5/18", 650, 120),
            box("示例上下文", 65, 148),
            box("我在复盘", 65, 232),
            box("当前用户", 65, 300),
            box("在干嘛", 65, 328),
        ],
    )

    records = _jsonl(output_jsonl)
    assert result.line_count == 3
    assert [(record["role"], record["speaker"], record["text"]) for record in records] == [
        ("self", "当前用户", "示例上下文"),
        ("target", "目标样本", "我在复盘"),
        ("self", "当前用户", "在干嘛"),
    ]
    assert {record["timestamp_ms"] for record in records} == {1747497600000}


def test_ocr_wechat_screenshots_rapidocr_skips_text_inside_media(
    tmp_path: Path,
) -> None:
    from PIL import Image

    from scripts.data.ocr_wechat_screenshots import ocr_wechat_screenshots, parse_args

    image_dir = tmp_path / "screens"
    image_dir.mkdir()
    image_path = image_dir / "shot_000001.png"
    image = Image.new("RGB", (700, 640), (242, 242, 242))
    for x in range(40, 170):
        for y in range(170, 230):
            image.putpixel((x, y), (20, 20, 20))
    image.save(image_path)
    output_jsonl = tmp_path / "ocr.jsonl"

    args = parse_args(
        [
            "--input-dir",
            str(image_dir),
            "--output-jsonl",
            str(output_jsonl),
            "--self-speaker-name",
            "当前用户",
            "--target-speaker-name",
            "目标样本",
            "--engine",
            "rapidocr",
            "--overwrite",
        ]
    )

    def box(text: str, x: float, y: float) -> tuple[list[list[float]], str, float]:
        return ([[x, y], [x + 120, y], [x + 120, y + 24], [x, y + 24]], text, 0.95)

    ocr_wechat_screenshots(
        args,
        rapidocr_reader=lambda _image: [
            box("目标样本", 55, 120),
            box("真正的文字消息", 55, 146),
            box("摸摸头", 82, 184),
        ],
    )

    records = _jsonl(output_jsonl)
    assert [(record["speaker"], record["text"]) for record in records] == [
        ("目标样本", "真正的文字消息")
    ]


def test_build_wechat_style_from_ocr_imports_only_target_and_pairs_self_context(
    tmp_path: Path,
) -> None:
    from scripts.data.build_wechat_style_from_ocr import build_style_from_ocr, parse_args

    ocr_jsonl = tmp_path / "ocr.jsonl"
    ocr_jsonl.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in [
                {
                    "role": "self",
                    "speaker": "当前用户",
                    "text": "示例上下文",
                    "source_image": "shot_000001.png",
                    "line_no": 3,
                    "timestamp_ms": 1747541820000,
                },
                {
                    "role": "target",
                    "speaker": "目标样本",
                    "text": "示例回复一",
                    "source_image": "shot_000001.png",
                    "line_no": 5,
                    "timestamp_ms": 1747541880000,
                },
                {
                    "role": "self",
                    "speaker": "当前用户",
                    "text": "在干嘛",
                    "source_image": "shot_000002.png",
                    "line_no": 3,
                    "timestamp_ms": 1747627200000,
                },
                {
                    "role": "target",
                    "speaker": "目标样本",
                    "text": "示例回复二",
                    "source_image": "shot_000002.png",
                    "line_no": 5,
                    "timestamp_ms": 1747627260000,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    raw_path = tmp_path / "raw.jsonl"
    manifest_path = tmp_path / "consent.json"
    samples_path = tmp_path / "samples.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    report_path = tmp_path / "report.json"

    args = parse_args(
        [
            "--ocr-jsonl",
            str(ocr_jsonl),
            "--self-speaker-name",
            "当前用户",
            "--target-speaker-name",
            "目标样本",
            "--persona-id",
            "demo_persona",
            "--subject-user-id",
            "1",
            "--consent-id",
            "consent-demo-persona-style",
            "--start-date",
            "2000-01-01",
            "--end-date",
            "2099-12-31",
            "--raw-jsonl",
            str(raw_path),
            "--consent-manifest",
            str(manifest_path),
            "--out-jsonl",
            str(samples_path),
            "--pairs-jsonl",
            str(pairs_path),
            "--report",
            str(report_path),
            "--overwrite",
        ]
    )

    output = build_style_from_ocr(args)

    raw_records = _jsonl(raw_path)
    sample_records = _jsonl(samples_path)
    pair_records = _jsonl(pairs_path)
    assert output.target_count == 2
    assert output.self_count == 2
    assert output.pair_count == 2
    assert [record["text"] for record in raw_records] == ["示例回复一", "示例回复二"]
    assert "示例上下文" not in {record["text"] for record in sample_records}
    assert pair_records[0]["self_text"] == "示例上下文"
    assert pair_records[0]["target_reply"] == "示例回复一"
    assert json.loads(report_path.read_text(encoding="utf-8"))["imported_count"] == 2


def test_build_wechat_style_from_ocr_pairs_by_message_order_when_timestamps_match(
    tmp_path: Path,
) -> None:
    from scripts.data.build_wechat_style_from_ocr import build_style_from_ocr, parse_args

    same_timestamp = 1747497600000
    ocr_jsonl = tmp_path / "ocr.jsonl"
    ocr_jsonl.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in [
                {
                    "role": "self",
                    "speaker": "当前用户",
                    "text": "以后你得少刷点抖音",
                    "source_image": "shot_000001.png",
                    "line_no": 3,
                    "timestamp_ms": same_timestamp,
                },
                {
                    "role": "target",
                    "speaker": "目标样本",
                    "text": "这几天不能熬夜",
                    "source_image": "shot_000001.png",
                    "line_no": 5,
                    "timestamp_ms": same_timestamp,
                },
                {
                    "role": "self",
                    "speaker": "当前用户",
                    "text": "你现在在干嘛",
                    "source_image": "shot_000002.png",
                    "line_no": 3,
                    "timestamp_ms": same_timestamp,
                },
                {
                    "role": "target",
                    "speaker": "目标样本",
                    "text": "示例回复二",
                    "source_image": "shot_000002.png",
                    "line_no": 5,
                    "timestamp_ms": same_timestamp,
                },
                {
                    "role": "target",
                    "speaker": "目标样本",
                    "text": "￥439 *",
                    "source_image": "shot_000002.png",
                    "line_no": 6,
                    "timestamp_ms": same_timestamp,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--ocr-jsonl",
            str(ocr_jsonl),
            "--self-speaker-name",
            "当前用户",
            "--target-speaker-name",
            "目标样本",
            "--persona-id",
            "demo_persona",
            "--subject-user-id",
            "1",
            "--consent-id",
            "consent-demo-persona-style",
            "--start-date",
            "2000-01-01",
            "--end-date",
            "2099-12-31",
            "--raw-jsonl",
            str(tmp_path / "raw.jsonl"),
            "--consent-manifest",
            str(tmp_path / "consent.json"),
            "--out-jsonl",
            str(tmp_path / "samples.jsonl"),
            "--pairs-jsonl",
            str(tmp_path / "pairs.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--overwrite",
        ]
    )

    output = build_style_from_ocr(args)
    pair_records = _jsonl(tmp_path / "pairs.jsonl")

    assert output.target_count == 2
    assert [record["text"] for record in _jsonl(tmp_path / "raw.jsonl")] == [
        "这几天不能熬夜",
        "示例回复二",
    ]
    assert [(record["self_text"], record["target_reply"]) for record in pair_records] == [
        ("以后你得少刷点抖音", "这几天不能熬夜"),
        ("你现在在干嘛", "示例回复二"),
    ]


def test_build_wechat_style_from_ocr_dedupes_overlapping_screenshots(
    tmp_path: Path,
) -> None:
    from scripts.data.build_wechat_style_from_ocr import build_style_from_ocr, parse_args

    ocr_jsonl = tmp_path / "ocr.jsonl"
    ocr_jsonl.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in [
                {
                    "role": "self",
                    "speaker": "当前用户",
                    "text": "你在干嘛",
                    "source_image": "shot_000010.png",
                    "line_no": 10,
                },
                {
                    "role": "target",
                    "speaker": "目标样本",
                    "text": "示例回复二",
                    "source_image": "shot_000010.png",
                    "line_no": 11,
                },
                {
                    "role": "target",
                    "speaker": "目标样本",
                    "text": "示例回复二",
                    "source_image": "shot_000011.png",
                    "line_no": 4,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--ocr-jsonl",
            str(ocr_jsonl),
            "--self-speaker-name",
            "当前用户",
            "--target-speaker-name",
            "目标样本",
            "--persona-id",
            "demo_persona",
            "--subject-user-id",
            "1",
            "--consent-id",
            "consent-demo-persona-style",
            "--start-date",
            "2000-01-01",
            "--end-date",
            "2099-12-31",
            "--raw-jsonl",
            str(tmp_path / "raw.jsonl"),
            "--consent-manifest",
            str(tmp_path / "consent.json"),
            "--out-jsonl",
            str(tmp_path / "samples.jsonl"),
            "--pairs-jsonl",
            str(tmp_path / "pairs.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--overwrite",
        ]
    )

    output = build_style_from_ocr(args)

    assert output.target_count == 1
    assert [record["text"] for record in _jsonl(tmp_path / "raw.jsonl")] == ["示例回复二"]


def test_build_wechat_style_from_ocr_fails_when_target_messages_are_missing(
    tmp_path: Path,
) -> None:
    import pytest

    from scripts.data.build_wechat_style_from_ocr import build_style_from_ocr, parse_args

    ocr_jsonl = tmp_path / "ocr.jsonl"
    ocr_jsonl.write_text(
        json.dumps(
            {
                "role": "self",
                "speaker": "当前用户",
                "text": "示例上下文",
                "source_image": "shot_000001.png",
                "line_no": 3,
                "timestamp_ms": 1747541820000,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--ocr-jsonl",
            str(ocr_jsonl),
            "--self-speaker-name",
            "当前用户",
            "--target-speaker-name",
            "目标样本",
            "--persona-id",
            "demo_persona",
            "--subject-user-id",
            "1",
            "--consent-id",
            "consent-demo-persona-style",
            "--start-date",
            "2000-01-01",
            "--end-date",
            "2099-12-31",
            "--raw-jsonl",
            str(tmp_path / "raw.jsonl"),
            "--consent-manifest",
            str(tmp_path / "consent.json"),
            "--out-jsonl",
            str(tmp_path / "samples.jsonl"),
            "--pairs-jsonl",
            str(tmp_path / "pairs.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    with pytest.raises(ValueError, match="no target speaker messages"):
        build_style_from_ocr(args)
