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


def test_final_docs_include_diagrams_demo_and_all_24_tutorials() -> None:
    architecture = Path("docs/architecture.md")
    demo_readme = Path("docs/demo/README.md")
    step24_tutorial = Path("docs/tutorials/step24_final_integration_demo.md")

    assert architecture.exists()
    assert demo_readme.exists()
    assert step24_tutorial.exists()

    architecture_text = architecture.read_text(encoding="utf-8")
    assert "PersonaAgent Architecture" in architecture_text
    assert "LangGraph Flow" in architecture_text
    assert "BotClient LiteIM Boundary" in architecture_text
    assert "RAG Collections" in architecture_text
    assert "Safety And Human Review" in architecture_text
    assert architecture_text.count("```mermaid") >= 5

    demo_text = demo_readme.read_text(encoding="utf-8")
    assert "Echo mode" in demo_text
    assert "Knowledge RAG" in demo_text
    assert "Eval report" in demo_text
    assert "scripts/demo/run_mock_demo.py" in demo_text

    tutorials = sorted(Path("docs/tutorials").glob("step*.md"))
    assert len(tutorials) == 24
    assert tutorials[-1].name == "step24_final_integration_demo.md"
