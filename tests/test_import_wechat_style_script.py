from __future__ import annotations

import json
from pathlib import Path


def test_wechat_style_import_writes_raw_processed_report_and_indexes_style(
    tmp_path: Path,
) -> None:
    from scripts.data.import_wechat_style import build_wechat_style_dataset, parse_args

    input_path = tmp_path / "wechat_copy.txt"
    input_path.write_text(
        "\n".join(
            [
                "2000-01-01 14:57\t目标样本\t示例回复一",
                "2000-01-01 15:00\t其他人\t这句不应该导入",
                "2000/01/02 09:30 目标样本: 示例回复二",
                "2000-12-31 23:59\t目标样本\t示例回复三",
                "2001-01-01 00:01\t目标样本\t这句超出范围",
                "[图片]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    raw_path = tmp_path / "raw.jsonl"
    manifest_path = tmp_path / "consent.json"
    samples_path = tmp_path / "samples.jsonl"
    report_path = tmp_path / "report.json"
    chroma_path = tmp_path / "chroma"

    args = parse_args(
        [
            "--input-text",
            str(input_path),
            "--speaker-name",
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
            "2000-12-31",
            "--raw-jsonl",
            str(raw_path),
            "--consent-manifest",
            str(manifest_path),
            "--out-jsonl",
            str(samples_path),
            "--report",
            str(report_path),
            "--index-chroma",
            str(chroma_path),
            "--overwrite",
        ]
    )

    output = build_wechat_style_dataset(args)

    raw_records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample_records = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert output.raw_count == 3
    assert output.imported_count == 3
    assert output.indexed_count == 3
    assert [record["text"] for record in raw_records] == [
        "示例回复一",
        "示例回复二",
        "示例回复三",
    ]
    assert {record["persona_id"] for record in sample_records} == {"demo_persona"}
    assert manifest["records"][0]["allowed_usage"] == ["style_simulation"]
    assert "hidden_impersonation" in manifest["records"][0]["forbidden_usage"]
    assert report["imported_count"] == 3
