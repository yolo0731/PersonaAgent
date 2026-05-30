from __future__ import annotations

import asyncio
import contextlib

from bot_client.connection.client import BotClient
from bot_client.runtime.config import BotClientSettings

# 连接生命周期管理层

# 这个类在后台创建 BotClient、连接、登录、等待断线，并在失败后按指数退避重试
class BotClientSupervisor:
    """Keeps the LiteIM BotClient connected without owning AgentService state."""

    def __init__(self, settings: BotClientSettings) -> None:
        self._settings = settings
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._client: BotClient | None = None
        self._last_error: Exception | None = None

    @property
    def client(self) -> BotClient | None:
        return self._client

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    async def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._client is not None:
            await self._client.close()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        delay = self._settings.reconnect_initial_delay_seconds
        while not self._stop_event.is_set():
            client = BotClient(self._settings)
            self._client = client
            try:
                await client.connect()
                await client.login()
                self._last_error = None
                delay = self._settings.reconnect_initial_delay_seconds
                await client.wait_disconnected()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = exc
            finally:
                await client.close()

            if self._stop_event.is_set():
                break
            await self._sleep_before_retry(delay)
            delay = self._next_delay(delay)

    async def _sleep_before_retry(self, delay: float) -> None:
        if delay <= 0:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)

    def _next_delay(self, delay: float) -> float:
        max_delay = self._settings.reconnect_max_delay_seconds
        if max_delay <= 0:
            return 0.0
        if delay <= 0:
            return max_delay
        return min(delay * 2.0, max_delay)
