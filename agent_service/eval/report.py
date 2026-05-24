from __future__ import annotations

import json
from pathlib import Path

from agent_service.eval.metrics import EvalReport

VARIANT_LABELS = {
    "no_rag": "No RAG",
    "knowledge": "Knowledge only",
    "knowledge_memory": "Knowledge + Memory",
    "knowledge_memory_style": "Knowledge + Memory + Style",
}


def write_json_report(report: EvalReport, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_markdown_report(report: EvalReport, path: str | Path, *, json_name: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown(report, json_name=json_name), encoding="utf-8")
    return output_path


def _markdown(report: EvalReport, *, json_name: str) -> str:
    metrics = report.metrics
    lines = [
        f"# PersonaAgent {report.mode.title()} Evaluation Report",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- JSON report: `{json_name}`",
        "",
        "## Sample Size",
        "",
        "| Dataset | Cases |",
        "| --- | ---: |",
        f"| RAG | {report.sample_size.rag} |",
        f"| Memory | {report.sample_size.memory} |",
        f"| Style | {report.sample_size.style} |",
        f"| Safety | {report.sample_size.safety} |",
        f"| LiteIM Integration | {report.sample_size.integration} |",
        f"| Total | {report.sample_size.total} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| RAG Hit@5 | {_percent(metrics.retrieval_hit_at_5)} |",
        f"| Memory Hit@5 | {_percent(metrics.memory_hit_at_5)} |",
        f"| Style Similarity | {_percent(metrics.style_similarity)} |",
        f"| Verbatim Leakage Rate | {_percent(metrics.verbatim_leakage_rate)} |",
        f"| Safety Violation Rate | {_percent(metrics.safety_violation_rate)} |",
        f"| Human Review Trigger Rate | {_percent(metrics.human_review_trigger_rate)} |",
        f"| Average latency | {metrics.average_latency_ms:.3f} ms |",
        f"| p95 latency | {metrics.p95_latency_ms:.3f} ms |",
        f"| Token cost per reply | ${metrics.token_cost_per_reply:.8f} |",
        (
            "| LiteIM integration success rate | "
            f"{_percent(metrics.liteim_integration_success_rate)} |"
        ),
        "",
        "## A/B Variants",
        "",
        "| Variant | Cases | Avg latency | p95 latency | Cost/reply | Success |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, variant in report.ab_variants.items():
        label = VARIANT_LABELS.get(key, key)
        lines.append(
            "| "
            f"{label} | {variant.sample_size} | {variant.average_latency_ms:.3f} ms | "
            f"{variant.p95_latency_ms:.3f} ms | ${variant.token_cost_per_reply:.8f} | "
            f"{_percent(variant.liteim_integration_success_rate)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"
