"""Background job scheduling.

A deliberately thin layer over ``asyncio`` tasks. Milestone 2 does not need a broker: the
work is short, the volume is low, and Celery plus Redis would add two processes and a
deployment story to solve a problem that does not exist yet.

The seam matters more than the implementation. Everything schedules work through
:class:`JobService`, so replacing this with a real queue later is a change to one class
rather than to every caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from app.core.logging import get_logger

logger = get_logger(__name__)


class JobService:
    """Runs background work with a bound on how much happens at once.

    Args:
        max_concurrent: How many jobs may run simultaneously. Without a bound, ten
            concurrent uploads of large spreadsheets would all be parsed at once and
            held in memory together.
    """

    def __init__(self, max_concurrent: int = 2) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        # Strong references to running tasks. asyncio only holds weak ones, so without
        # this a task can be garbage-collected mid-flight and simply vanish.
        self._tasks: set[asyncio.Task[None]] = set()

    def schedule(self, name: str, work: Callable[[], Awaitable[object]]) -> asyncio.Task[None]:
        """Start background work and return its task.

        Args:
            name: Short label used in logs, e.g. ``"ingest:<document-id>"``.
            work: Zero-argument coroutine function to run.

        Returns:
            The created task. Tests await it; production code does not.
        """
        task = asyncio.create_task(self._run(name, work), name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(self, name: str, work: Callable[[], Awaitable[object]]) -> None:
        """Execute one job, holding the concurrency slot for its duration."""
        async with self._semaphore:
            try:
                await work()
            except Exception:
                # The job itself is responsible for recording its own failure; this is
                # the last line of defence so a background failure is never invisible.
                logger.exception("Background job failed name=%s", name)

    async def wait_for_all(self, timeout: float = 30.0) -> None:  # noqa: ASYNC109
        """Wait for outstanding jobs to finish.

        Used at shutdown so in-flight work is not abandoned, and by tests that need the
        background pipeline to have completed before asserting.
        """
        pending = list(self._tasks)
        if not pending:
            return

        _done, still_running = await asyncio.wait(pending, timeout=timeout)
        if still_running:
            logger.warning(
                "%d background job(s) did not finish within %.0fs", len(still_running), timeout
            )

    @property
    def active_count(self) -> int:
        """How many jobs are currently tracked."""
        return len(self._tasks)
