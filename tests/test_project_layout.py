from __future__ import annotations

import re
from pathlib import Path

OLD_BOT_CLIENT_MODULES = (
    "liteim_protocol",
    "frame_decoder",
    "protocol_builders",
    "protocol_parser",
    "bot_client",
    "message_handler",
    "message_state",
    "echo",
    "agent_api",
    "friend_policy",
    "config",
    "runtime",
    "supervisor",
)
OLD_BOT_CLIENT_IMPORTS = tuple(f"bot_client.{module}" for module in OLD_BOT_CLIENT_MODULES)
OLD_BOT_CLIENT_IMPORT_PATTERNS = tuple(
    re.compile(rf"(?<![\w.]){re.escape(import_path)}(?![\w.])")
    for import_path in OLD_BOT_CLIENT_IMPORTS
)


def test_runtime_data_defaults_are_grouped_by_purpose() -> None:
    from agent_service.config import Settings
    from bot_client.runtime.config import BotClientSettings

    settings = Settings(_env_file=None)
    bot_settings = BotClientSettings(_env_file=None)

    assert settings.agent_state_db_path == "data/state/agent_state/state.sqlite3"
    assert settings.memory_db_path == "data/state/memory/memory.sqlite3"
    assert settings.chroma_path == "data/vector/chroma"
    assert settings.knowledge_docs_path == "data/knowledge_docs"
    assert bot_settings.bot_state_path == "data/state/bot_state/state.json"


def test_new_script_packages_expose_runtime_demo_and_data_entrypoints() -> None:
    from scripts.data import (
        build_wechat_style_from_ocr,
        capture_wechat_chat,
        import_wechat_style,
        ocr_wechat_screenshots,
    )
    from scripts.demo import run_mock_demo
    from scripts.runtime import run_bot_client

    assert callable(run_bot_client.parse_args)
    assert callable(run_mock_demo.main)
    assert callable(import_wechat_style.parse_args)
    assert callable(capture_wechat_chat.parse_args)
    assert callable(ocr_wechat_screenshots.parse_args)
    assert callable(build_wechat_style_from_ocr.parse_args)


def test_expected_runtime_directories_are_explicitly_scaffolded() -> None:
    root = Path("data")

    assert (root / "runtime").is_dir()
    assert (root / "state" / "agent_state").is_dir()
    assert (root / "state" / "bot_state").is_dir()
    assert (root / "state" / "memory").is_dir()
    assert (root / "vector" / "chroma").is_dir()
    assert (root / "authorized_style_records" / "raw").is_dir()
    assert (root / "authorized_style_records" / "processed").is_dir()
    assert (root / "knowledge_docs").is_dir()


def test_env_example_has_unique_runtime_keys() -> None:
    env_example = Path(".env.example")
    keys: list[str] = []
    for line in env_example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.append(stripped.split("=", maxsplit=1)[0])

    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    assert duplicates == []


def test_readme_describes_current_implemented_agent_not_old_mock_stage() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Python BotClient plus FastAPI AgentService" in readme
    assert "AgentService `/chat` API" in readme
    assert "mock reply handler" not in readme


def test_bot_client_root_only_contains_responsibility_packages() -> None:
    bot_client_root = Path("bot_client")
    allowed_entries = {
        "__init__.py",
        "access",
        "agent",
        "connection",
        "messages",
        "protocol",
        "runtime",
    }

    actual_entries = {
        entry.name
        for entry in bot_client_root.iterdir()
        if entry.name != "__pycache__"
    }

    assert actual_entries == allowed_entries

    for package in allowed_entries - {"__init__.py"}:
        assert (bot_client_root / package / "__init__.py").is_file()


def test_current_sources_do_not_reference_old_bot_client_import_paths() -> None:
    paths_to_scan = [
        Path("README.md"),
        Path("agent_service"),
        Path("bot_client"),
        Path("scripts"),
        Path("tests"),
    ]
    workspace_project = Path("../PROJECT.md")
    if workspace_project.exists():
        paths_to_scan.append(workspace_project)

    files: list[Path] = []
    for path in paths_to_scan:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix in {".md", ".py"}
            )

    violations: list[str] = []
    current_test = Path(__file__).relative_to(Path.cwd())
    for file_path in sorted(files):
        if file_path == current_test:
            continue
        text = file_path.read_text(encoding="utf-8")
        for old_import, pattern in zip(
            OLD_BOT_CLIENT_IMPORTS, OLD_BOT_CLIENT_IMPORT_PATTERNS, strict=True
        ):
            if pattern.search(text):
                violations.append(f"{file_path}: {old_import}")

    assert violations == []
