# ruff: noqa: E402
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CommandRunner = Callable[[list[str]], str]
ToolLookup = Callable[[str], str | None]
Sleeper = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class CaptureResult:
    images: list[Path]
    output_dir: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture an opened WeChat chat window while scrolling upward.",
    )
    parser.add_argument("--window-id", help="X11 window id from xdotool search.")
    parser.add_argument(
        "--window-name",
        default="目标样本",
        help="Window name used when --window-id is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "data/authorized_style_records/raw/wechat_screenshots/"
            "demo_persona_sample_range"
        ),
    )
    parser.add_argument("--max-shots", type=int, default=120)
    parser.add_argument("--interval-seconds", type=float, default=0.35)
    parser.add_argument("--scroll-clicks", type=int, default=5)
    parser.add_argument("--scroll-x", type=int, default=500)
    parser.add_argument("--scroll-y", type=int, default=500)
    parser.add_argument(
        "--screenshot-tool",
        choices=("gnome-screenshot", "import"),
        default="gnome-screenshot",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Continue after the largest existing shot_*.png index in output-dir.",
    )
    return parser.parse_args(argv)


def capture_chat_screenshots(
    args: argparse.Namespace,
    *,
    command_runner: CommandRunner | None = None,
    tool_lookup: ToolLookup = shutil.which,
    sleeper: Sleeper = time.sleep,
) -> CaptureResult:
    runner = command_runner or _run_command
    if args.max_shots <= 0:
        raise ValueError("max-shots must be positive")
    if args.scroll_clicks <= 0:
        raise ValueError("scroll-clicks must be positive")
    _require_tool("xdotool", tool_lookup)
    _require_tool(args.screenshot_tool, tool_lookup)

    output_dir = Path(args.output_dir)
    start_index = _prepare_output_dir(
        output_dir,
        overwrite=bool(args.overwrite),
        append=bool(args.append),
    )
    window_id = args.window_id or _find_window_id(args.window_name, runner)

    runner(["xdotool", "windowactivate", "--sync", window_id])
    runner(
        [
            "xdotool",
            "mousemove",
            "--window",
            window_id,
            str(args.scroll_x),
            str(args.scroll_y),
        ]
    )

    images: list[Path] = []
    for index in range(start_index, start_index + args.max_shots):
        image_path = output_dir / f"shot_{index:06d}.png"
        _capture_image(
            image_path,
            window_id=window_id,
            screenshot_tool=args.screenshot_tool,
            command_runner=runner,
        )
        images.append(image_path)
        runner(["xdotool", "click", "--repeat", str(args.scroll_clicks), "4"])
        sleeper(float(args.interval_seconds))
    return CaptureResult(images=images, output_dir=output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = capture_chat_screenshots(args)
    except Exception as exc:
        print(f"WeChat capture failed: {exc}", file=sys.stderr)
        return 1
    print(f"WeChat capture complete images={len(result.images)} dir={result.output_dir}")
    return 0


def _capture_image(
    image_path: Path,
    *,
    window_id: str,
    screenshot_tool: str,
    command_runner: CommandRunner,
) -> None:
    if screenshot_tool == "gnome-screenshot":
        command_runner(["gnome-screenshot", "-f", str(image_path)])
        return
    command_runner(["import", "-window", window_id, str(image_path)])


def _find_window_id(window_name: str, command_runner: CommandRunner) -> str:
    output = command_runner(["xdotool", "search", "--name", window_name]).strip()
    for line in output.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    raise RuntimeError(f"cannot find window by name: {window_name}")


def _prepare_output_dir(path: Path, *, overwrite: bool, append: bool) -> int:
    if overwrite and append:
        raise ValueError("--overwrite and --append cannot be used together")
    if path.exists() and any(path.iterdir()) and not overwrite and not append:
        raise FileExistsError(f"{path} is not empty; pass --overwrite to replace it")
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for child in path.glob("shot_*.png"):
            child.unlink()
        return 1
    if append:
        return _next_shot_index(path.glob("shot_*.png"))
    return 1


def _next_shot_index(paths: Sequence[Path]) -> int:
    max_index = 0
    for path in paths:
        try:
            max_index = max(max_index, int(path.stem.removeprefix("shot_")))
        except ValueError:
            continue
    return max_index + 1


def _require_tool(name: str, tool_lookup: ToolLookup) -> None:
    if tool_lookup(name) is None:
        raise RuntimeError(f"required tool not found in PATH: {name}")


def _run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


if __name__ == "__main__":
    raise SystemExit(main())
