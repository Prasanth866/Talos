from __future__ import annotations

import asyncio
import contextlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.agent.dispatcher import create_default_dispatcher
from src.agent.llm_client import BaseLLMClient, HTTPLLMClient, MockLLMClient
from src.agent.loop import ReasoningLoop
from src.agent.models import TrajectoryStatus
from src.agent.pipeline import execute_workspace_task
from src.api.schemas.events import AgentEvent, ErrorEvent
from src.core.config import ROOT_DIR, Settings, get_settings
from src.core.logging import get_logger
from src.db import repository
from src.workspace.manager import WorkspaceManager

logger = get_logger(__name__)

LoopFactory = Callable[[], ReasoningLoop]


def create_default_llm_client(
    settings: Settings | None = None,
    llm_client_override: BaseLLMClient | None = None,
) -> BaseLLMClient:
    """Creates a default BaseLLMClient based on application configuration."""
    if llm_client_override is not None:
        return llm_client_override

    cfg = settings or get_settings()
    if cfg.llm_api_key.get_secret_value():
        return HTTPLLMClient(
            api_key=cfg.llm_api_key.get_secret_value(),
            base_url=cfg.llm_base_url,
            model=cfg.llm_model,
            timeout_seconds=cfg.llm_timeout_seconds,
            max_retries=cfg.llm_max_retries,
            initial_delay=cfg.llm_retry_initial_delay,
            backoff_factor=cfg.llm_retry_backoff_factor,
        )

    return MockLLMClient(
        responses=[
            {
                "thought": "Processing received task in development mode.",
                "final_answer": "Task executed successfully (mock mode).",
            }
        ]
    )


def create_default_loop(
    settings: Settings | None = None,
    llm_client_override: BaseLLMClient | None = None,
) -> ReasoningLoop:
    """Creates a default ReasoningLoop instance using app configuration."""
    cfg = settings or get_settings()
    dispatcher = create_default_dispatcher(ROOT_DIR)
    llm_client = create_default_llm_client(cfg, llm_client_override)

    return ReasoningLoop(
        llm_client=llm_client,
        dispatcher=dispatcher,
        max_steps=cfg.llm_max_steps,
    )


@dataclass
class TaskItem:
    """Represents an enqueued task item for worker processing."""

    task_id: str
    task: str
    metadata: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None
    max_cost_usd: float | None = None


class TaskManager:
    """Manages the bounded task queue, asyncio.TaskGroup worker pool,

    subscriber event streaming, and graceful shutdown draining.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        loop_factory: LoopFactory | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        workspace_manager: WorkspaceManager | None = None,
        max_retained_task_events: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.loop_factory: LoopFactory = loop_factory or (
            lambda: create_default_loop(self.settings)
        )
        self.session_factory = session_factory
        self.workspace_manager = workspace_manager
        self.queue: asyncio.Queue[TaskItem | None] = asyncio.Queue(
            maxsize=self.settings.task_queue_max_size
        )
        self.worker_concurrency: int = self.settings.worker_concurrency
        self.drain_timeout_seconds: float = self.settings.shutdown_drain_timeout_seconds
        self.max_retained_task_events: int = (
            max_retained_task_events
            if max_retained_task_events is not None
            else self.settings.max_retained_task_events
        )
        self._shutting_down: bool = False
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._subscribers: dict[str, list[asyncio.Queue[AgentEvent | None]]] = {}
        self._task_events: dict[str, list[AgentEvent]] = {}
        self._task_completed: OrderedDict[str, bool] = OrderedDict()
        self._subscribers_lock = asyncio.Lock()
        self._active_tasks: set[str] = set()

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    @property
    def queue_size(self) -> int:
        return self.queue.qsize()

    @property
    def queue_capacity(self) -> int:
        return self.queue.maxsize

    @property
    def active_task_count(self) -> int:
        return len(self._active_tasks)

    def submit(
        self,
        task_id: str,
        task: str,
        metadata: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> bool:
        """Submits a new task to the bounded queue.

        Returns True if successfully queued, False if queue is full or shutting down.
        """
        if self._shutting_down:
            logger.warning(
                "task_submission_rejected_shutting_down",
                task_id=task_id,
            )
            return False

        try:
            item = TaskItem(
                task_id=task_id,
                task=task,
                metadata=metadata or {},
                max_tokens=max_tokens,
                max_cost_usd=max_cost_usd,
            )
            self.queue.put_nowait(item)
            logger.info(
                "task_enqueued",
                task_id=task_id,
                queue_size=self.queue.qsize(),
                max_size=self.queue.maxsize,
            )
            return True
        except asyncio.QueueFull:
            logger.warning(
                "task_queue_full_rejected",
                task_id=task_id,
                queue_size=self.queue.qsize(),
                max_size=self.queue.maxsize,
            )
            return False

    async def subscribe(self, task_id: str) -> asyncio.Queue[AgentEvent | None]:
        """Subscribes to events for a specific task_id, replaying existing history."""
        sub_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        async with self._subscribers_lock:
            for past_event in self._task_events.get(task_id, []):
                sub_queue.put_nowait(past_event)

            if self._task_completed.get(task_id, False):
                sub_queue.put_nowait(None)
            else:
                self._subscribers.setdefault(task_id, []).append(sub_queue)

        return sub_queue

    async def unsubscribe(
        self,
        task_id: str,
        sub_queue: asyncio.Queue[AgentEvent | None],
    ) -> None:
        """Unsubscribes from events for a specific task_id."""
        async with self._subscribers_lock:
            if task_id in self._subscribers:
                with (
                    structlog.contextvars.bound_contextvars(task_id=task_id),
                    contextlib.suppress(ValueError),
                ):
                    self._subscribers[task_id].remove(sub_queue)
                if not self._subscribers[task_id]:
                    del self._subscribers[task_id]

    async def broadcast_event(self, task_id: str, event: AgentEvent | None) -> None:
        """Broadcasts an event (or sentinel None) to all listeners for task_id."""
        async with self._subscribers_lock:
            if event is not None:
                self._task_events.setdefault(task_id, []).append(event)
            else:
                self._task_completed[task_id] = True
                while len(self._task_completed) > self.max_retained_task_events:
                    oldest_task_id, _ = self._task_completed.popitem(last=False)
                    self._task_events.pop(oldest_task_id, None)

            subscribers = list(self._subscribers.get(task_id, []))

        for sub_queue in subscribers:
            try:
                sub_queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "subscriber_queue_full_dropping_event",
                    task_id=task_id,
                )

    async def prune_task(self, task_id: str) -> None:
        """Explicitly cleans up in-memory events and completed status for a task."""
        async with self._subscribers_lock:
            self._task_events.pop(task_id, None)
            self._task_completed.pop(task_id, None)

    async def _worker(self, worker_id: int) -> None:
        """Worker loop pulling tasks from the queue and running reasoning loops."""
        logger.debug("worker_started", worker_id=worker_id)
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                logger.debug("worker_received_sentinel_exiting", worker_id=worker_id)
                break

            task_id = item.task_id
            self._active_tasks.add(task_id)
            structlog.contextvars.bind_contextvars(task_id=task_id)
            logger.info("worker_task_started", worker_id=worker_id, task_id=task_id)

            if self.session_factory is not None:
                async with self.session_factory() as session:
                    await repository.mark_running(session, task_id)

            async def on_event(evt: AgentEvent, tid: str = task_id) -> None:
                await self.broadcast_event(tid, evt)

            try:
                repo_url = (item.metadata or {}).get("repo_url")
                if repo_url and isinstance(repo_url, str):
                    wm = self.workspace_manager or WorkspaceManager()
                    llm_client = create_default_llm_client(self.settings)
                    trajectory, _ = await execute_workspace_task(
                        task=item.task,
                        repo_url=repo_url,
                        workspace_manager=wm,
                        llm_client=llm_client,
                        max_steps=self.settings.llm_max_steps,
                        on_event=on_event,
                        task_id=task_id,
                        database_session_factory=self.session_factory,
                    )
                else:
                    loop = self.loop_factory()
                    tracker = getattr(loop.llm_client, "token_tracker", None)
                    if tracker is not None:
                        if item.max_tokens is not None:
                            tracker.max_tokens = item.max_tokens
                        if item.max_cost_usd is not None:
                            tracker.max_cost_usd = item.max_cost_usd

                    trajectory = await loop.run(
                        task=item.task,
                        metadata=item.metadata,
                        on_event=on_event,
                        task_id=task_id,
                    )

                if self.session_factory is not None:
                    async with self.session_factory() as session:
                        if trajectory.status == TrajectoryStatus.COMPLETED:
                            await repository.mark_completed(
                                session=session,
                                task_id=task_id,
                                result=trajectory.final_answer,
                                prompt_tokens=trajectory.total_tokens.prompt_tokens,
                                completion_tokens=trajectory.total_tokens.completion_tokens,
                                total_tokens=trajectory.total_tokens.total_tokens,
                                total_cost_usd=trajectory.total_cost_usd,
                                duration_seconds=trajectory.total_duration_seconds,
                            )
                        else:
                            await repository.mark_failed(
                                session=session,
                                task_id=task_id,
                                error=(
                                    trajectory.error
                                    or f"Task failed with status: {trajectory.status}"
                                ),
                                result=trajectory.final_answer,
                                prompt_tokens=trajectory.total_tokens.prompt_tokens,
                                completion_tokens=trajectory.total_tokens.completion_tokens,
                                total_tokens=trajectory.total_tokens.total_tokens,
                                total_cost_usd=trajectory.total_cost_usd,
                                duration_seconds=trajectory.total_duration_seconds,
                            )

                logger.info(
                    "worker_task_completed",
                    worker_id=worker_id,
                    task_id=task_id,
                )
            except asyncio.CancelledError:
                logger.warning(
                    "worker_task_cancelled",
                    worker_id=worker_id,
                    task_id=task_id,
                )
                if self.session_factory is not None:
                    async with self.session_factory() as session:
                        await repository.mark_failed(
                            session=session,
                            task_id=task_id,
                            error="Worker task was cancelled",
                        )
                raise
            except Exception as exc:
                logger.exception(
                    "worker_task_error",
                    worker_id=worker_id,
                    task_id=task_id,
                    error=str(exc),
                )
                if self.session_factory is not None:
                    async with self.session_factory() as session:
                        await repository.mark_failed(
                            session=session,
                            task_id=task_id,
                            error=f"Worker processing error: {exc}",
                        )
                err_event = ErrorEvent(
                    error=f"Worker processing error: {exc}",
                    task_id=task_id,
                )
                await self.broadcast_event(task_id, err_event)
            finally:
                self._active_tasks.discard(task_id)
                await self.broadcast_event(task_id, None)
                self.queue.task_done()
                structlog.contextvars.unbind_contextvars("task_id")

    def start(self, task_group: asyncio.TaskGroup | None = None) -> None:
        """Launches worker tasks inside TaskGroup or via asyncio.create_task."""
        for i in range(self.worker_concurrency):
            if task_group is not None:
                task = task_group.create_task(
                    self._worker(worker_id=i + 1),
                    name=f"worker-{i + 1}",
                )
            else:
                task = asyncio.create_task(
                    self._worker(worker_id=i + 1),
                    name=f"worker-{i + 1}",
                )
            self._worker_tasks.append(task)
        logger.info(
            "worker_pool_started",
            workers=self.worker_concurrency,
            queue_capacity=self.queue_capacity,
        )

    async def stop(self) -> None:
        """Initiates graceful shutdown:

        1. Mark as shutting down (rejecting new tasks)
        2. Unblock all WebSocket subscriber queues
        3. Enqueue sentinel None values for all workers
        4. Wait for worker tasks to drain up to drain_timeout_seconds
        """
        self._shutting_down = True
        logger.info(
            "worker_pool_stopping",
            active_tasks=self.active_task_count,
            queued_tasks=self.queue_size,
        )

        if self.worker_concurrency == 0:
            return

        async with self._subscribers_lock:
            for sub_list in self._subscribers.values():
                for sub_queue in sub_list:
                    with contextlib.suppress(Exception):
                        sub_queue.put_nowait(None)

        for _ in range(self.worker_concurrency):
            with contextlib.suppress(Exception):
                self.queue.put_nowait(None)

        if self._worker_tasks:
            _done, pending = await asyncio.wait(
                self._worker_tasks,
                timeout=self.drain_timeout_seconds,
            )
            if pending:
                logger.warning(
                    "worker_pool_drain_timed_out_cancelling",
                    timeout_seconds=self.drain_timeout_seconds,
                    pending_workers=len(pending),
                )
                for worker_task in pending:
                    worker_task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        self._worker_tasks.clear()
        logger.info("worker_pool_stopped_successfully")
