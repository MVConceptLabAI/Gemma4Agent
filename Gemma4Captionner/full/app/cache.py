"""
Tiny process-local memo cache.

The judging harness cannot re-use anything between submissions (containers
are ephemeral), but WITHIN a single run of the container we often see the
same video referenced multiple times, or repeated (facts, style) pairs
when the style layer retries. Caching those cuts wall-time and tokens
spent.

Deliberately NOT persisted to disk — the guide bans hardcoded / cached
answers to specific inputs and evaluates on unseen variants. In-memory
memoisation for the current run only is fine.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Callable, Coroutine


class AsyncMemoCache:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    async def get_or_compute(
        self,
        key: str,
        compute: Callable[[], Coroutine[Any, Any, Any]],
    ) -> Any:
        if key in self._store:
            return self._store[key]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._store:  # someone else filled it while we waited
                return self._store[key]
            value = await compute()
            self._store[key] = value
            return value


# Module-level singleton — small enough to be free, useful enough to matter.
CACHE = AsyncMemoCache()
