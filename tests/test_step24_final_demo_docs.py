from __future__ import annotations

import json
from pathlib import Path

from scripts.demo.run_mock_demo import run_mock_demo


def test_mock_demo_script_writes_required_scenario_outputs(tmp_path: Path) -> None:
    output = run_mock_demo(output_dir=tmp_path)

    payload = json.loads(output.json_path.read_text(encoding="utf-8"))
    transcript = output.transcript_path.read_text(encoding="utf-8")
    scenario_names = {scenario["name"] for scenario in payload["scenarios"]}

    assert scenario_names == {
        "echo_mode",
        "knowledge_rag",
        "memory_rag",
        "authorized_style_rag",
        "tool_calling",
        "safety_block",
        "human_review",
        "eval_report",
    }
    assert "Echo mode" in transcript
    assert "Knowledge RAG" in transcript
    assert "Human Review" in transcript
    assert payload["eval_report"]["sample_size"]["total"] > 0
