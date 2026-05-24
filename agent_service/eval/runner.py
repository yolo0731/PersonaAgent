from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from agent_service.eval.cases import load_eval_datasets
from agent_service.eval.metrics import evaluate_datasets
from agent_service.eval.report import write_json_report, write_markdown_report


class EvalMode(StrEnum):
    MOCK = "mock"
    REAL = "real"


class EvalOptions(BaseModel):
    mode: EvalMode = EvalMode.MOCK
    prompt_token_cost_per_1k: float = Field(default=0.0, ge=0.0)
    completion_token_cost_per_1k: float = Field(default=0.0, ge=0.0)
    openai_api_key: str | None = None


class EvalRunOutput(BaseModel):
    json_path: Path
    markdown_path: Path


def run_eval_suite(
    *,
    datasets_dir: str | Path,
    output_dir: str | Path,
    options: EvalOptions | None = None,
) -> EvalRunOutput:
    effective_options = options or EvalOptions()
    if effective_options.mode == EvalMode.REAL:
        _require_real_eval_key(effective_options)

    datasets = load_eval_datasets(datasets_dir)
    report = evaluate_datasets(
        datasets,
        mode=effective_options.mode.value,
        prompt_token_cost_per_1k=effective_options.prompt_token_cost_per_1k,
        completion_token_cost_per_1k=effective_options.completion_token_cost_per_1k,
    )
    root = Path(output_dir)
    json_path = root / f"{effective_options.mode.value}_eval_report.json"
    markdown_path = root / f"{effective_options.mode.value}_eval_report.md"
    write_json_report(report, json_path)
    write_markdown_report(report, markdown_path, json_name=json_path.name)
    return EvalRunOutput(json_path=json_path, markdown_path=markdown_path)


def _require_real_eval_key(options: EvalOptions) -> None:
    if options.openai_api_key or os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError("OPENAI_API_KEY is required for real eval mode")
