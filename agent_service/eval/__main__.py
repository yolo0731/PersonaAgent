from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent_service.eval.runner import EvalMode, EvalOptions, run_eval_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PersonaAgent evaluation suite.")
    parser.add_argument("--mode", choices=[mode.value for mode in EvalMode], default="mock")
    parser.add_argument("--datasets-dir", default="eval/datasets")
    parser.add_argument("--output-dir", default="eval/reports")
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument(
        "--prompt-token-cost-per-1k",
        type=float,
        default=float(os.getenv("EVAL_PROMPT_TOKEN_COST_PER_1K", "0")),
    )
    parser.add_argument(
        "--completion-token-cost-per-1k",
        type=float,
        default=float(os.getenv("EVAL_COMPLETION_TOKEN_COST_PER_1K", "0")),
    )
    args = parser.parse_args()

    output = run_eval_suite(
        datasets_dir=Path(args.datasets_dir),
        output_dir=Path(args.output_dir),
        options=EvalOptions(
            mode=EvalMode(args.mode),
            prompt_token_cost_per_1k=args.prompt_token_cost_per_1k,
            completion_token_cost_per_1k=args.completion_token_cost_per_1k,
            real_eval_confirm=os.getenv("REAL_EVAL_CONFIRM") == "1",
            max_cases=args.max_cases,
            concurrency=args.concurrency,
            resume=args.resume,
            fail_fast=args.fail_fast,
            sample_seed=args.sample_seed,
        ),
    )
    print(f"Wrote {output.json_path}")
    print(f"Wrote {output.markdown_path}")
    if output.results_path.exists():
        print(f"Wrote {output.results_path}")


if __name__ == "__main__":
    main()
