# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot_client.runtime.app import AgentBotRuntime, EchoBotRuntime
from bot_client.runtime.config import BotClientSettings


class BotRuntime(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PersonaAgent BotClient as a normal LiteIM user.",
    )
    parser.add_argument(
        "--mode",
        choices=("agent", "echo"),
        default="agent",
        help="agent calls AgentService /chat; echo only repeats private messages.",
    )
    parser.add_argument("--username", help="LiteIM bot username, for example persona_bot.")
    parser.add_argument("--password", help="LiteIM bot password.")
    parser.add_argument("--nickname", help="LiteIM bot nickname used by register flows.")
    parser.add_argument("--liteim-host", help="LiteIM server host.")
    parser.add_argument("--liteim-port", type=int, help="LiteIM server port.")
    parser.add_argument("--agent-service-url", help="AgentService base URL.")
    parser.add_argument("--state-path", help="Local BotClient state JSON path.")
    parser.add_argument("--offline-message-limit", type=int, help="Offline pull limit.")
    parser.add_argument("--allowed-user-ids", help="Comma-separated allowlisted user ids.")
    parser.add_argument("--allowed-usernames", help="Comma-separated allowlisted usernames.")
    parser.add_argument(
        "--env-file",
        default=str(PROJECT_ROOT / ".env"),
        help="Environment file to load; pass an empty string to disable file loading.",
    )
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> BotClientSettings:
    values: dict[str, Any] = {}
    _put_if_present(values, "bot_username", args.username)
    _put_if_present(values, "bot_password", args.password)
    _put_if_present(values, "bot_nickname", args.nickname)
    _put_if_present(values, "liteim_host", args.liteim_host)
    _put_if_present(values, "liteim_port", args.liteim_port)
    _put_if_present(values, "agent_service_url", args.agent_service_url)
    _put_if_present(values, "bot_state_path", args.state_path)
    _put_if_present(values, "offline_message_limit", args.offline_message_limit)
    _put_if_present(values, "allowed_user_ids", args.allowed_user_ids)
    _put_if_present(values, "allowed_usernames", args.allowed_usernames)
    env_file = args.env_file if args.env_file else None
    return BotClientSettings(_env_file=env_file, **values)


def build_runtime(args: argparse.Namespace, settings: BotClientSettings) -> BotRuntime:
    if args.mode == "echo":
        return EchoBotRuntime(settings)
    return AgentBotRuntime(settings)


async def run(args: argparse.Namespace) -> None:
    settings = build_settings(args)
    runtime = build_runtime(args, settings)
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    await runtime.start()
    _print_startup(args, settings)
    try:
        await stop_event.wait()
    finally:
        await runtime.stop()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"PersonaAgent BotClient failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _put_if_present(values: dict[str, Any], key: str, value: object | None) -> None:
    if value is not None:
        values[key] = value


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            return


def _print_startup(args: argparse.Namespace, settings: BotClientSettings) -> None:
    print(
        "PersonaAgent BotClient started "
        f"mode={args.mode} "
        f"user={settings.bot_username} "
        f"liteim={settings.liteim_host}:{settings.liteim_port} "
        f"agent_service={settings.agent_service_url}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
