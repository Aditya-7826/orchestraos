"""Runs each goal as a background task and keeps its events so clients can (re)connect.

Why not stream straight from the POST request? A run takes minutes on a laptop,
and Streamlit re-runs its script on every interaction, which would drop the
connection. Here the run keeps going on its own; clients read its events from
any index with `events(thread_id, after=n)`. Events live in memory, but the
run's *state* is in the SQLite checkpointer, so after an API restart a stopped
run can still be resumed with `resume(thread_id)`.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph

from orchestraos.api.events import Event, to_events

logger = logging.getLogger(__name__)

RunStatus = Literal["running", "completed", "interrupted", "failed"]


class RunNotFound(KeyError):
    pass


@dataclass
class Run:
    thread_id: str
    status: RunStatus = "running"
    events: list[Event] = field(default_factory=list)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task | None = None

    async def emit(self, event: Event) -> None:
        async with self.changed:
            self.events.append(event)
            self.changed.notify_all()


class RunManager:
    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph
        self.runs: dict[str, Run] = {}

    @staticmethod
    def config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def start(self, goal: str) -> Run:
        run = Run(thread_id=uuid.uuid4().hex[:12])
        self._launch(run, {"goal": goal})
        return run

    async def resume(self, thread_id: str, command: Any = None) -> Run:
        """Continue a stopped or interrupted run. `command` is a LangGraph Command (approvals)."""
        existing = self.runs.get(thread_id)
        if existing and existing.status == "running":
            raise ValueError(f"run {thread_id} is still running")
        snapshot = await self.graph.aget_state(self.config(thread_id))
        if not snapshot.values:
            raise RunNotFound(thread_id)
        if not snapshot.next:
            raise ValueError(f"run {thread_id} has already finished")
        # Keep earlier events so a client that reconnects still sees the whole run.
        run = existing or Run(thread_id=thread_id)
        run.status = "running"
        self._launch(run, command)
        return run

    def _launch(self, run: Run, graph_input: Any) -> None:
        self.runs[run.thread_id] = run
        run.task = asyncio.create_task(self._drive(run, graph_input))

    async def _drive(self, run: Run, graph_input: Any) -> None:
        config = self.config(run.thread_id)
        try:
            async for namespace, mode, chunk in self.graph.astream(
                graph_input, config, stream_mode=["updates", "messages"], subgraphs=True
            ):
                for event in to_events(namespace, mode, chunk):
                    await run.emit(event)
            snapshot = await self.graph.aget_state(config)
            run.status = "interrupted" if snapshot.next else "completed"
        except Exception as exc:
            logger.exception("run %s failed", run.thread_id)
            run.status = "failed"
            await run.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        await run.emit({"type": "done", "status": run.status})

    async def events(self, thread_id: str, after: int = 0) -> AsyncIterator[tuple[int, Event]]:
        """Yield (index, event) from `after` on, waiting for new ones until the run stops."""
        run = self.runs.get(thread_id)
        if run is None:
            raise RunNotFound(thread_id)
        index = after
        while True:
            async with run.changed:
                await run.changed.wait_for(lambda seen=index: len(run.events) > seen)
                batch = run.events[index:]
            for event in batch:
                yield index, event
                index += 1
                if event["type"] == "done":
                    return

    async def snapshot(self, thread_id: str) -> dict:
        """The run's current state from the checkpointer (works after an API restart too)."""
        snapshot = await self.graph.aget_state(self.config(thread_id))
        values = snapshot.values
        if not values:
            raise RunNotFound(thread_id)
        live = self.runs.get(thread_id)
        if live and live.status == "running":
            status: RunStatus = "running"
        elif snapshot.next:
            status = live.status if live and live.status == "failed" else "interrupted"
        else:
            status = "completed"
        return {
            "thread_id": thread_id,
            "status": status,
            "goal": values.get("goal"),
            "plan": values.get("plan", []),
            "results": values.get("results", []),
            "final_answer": values.get("final_answer"),
            "sources": values.get("sources", []),
            "artifacts": values.get("artifacts", []),
            "pending": [i.value for task in snapshot.tasks for i in task.interrupts],
        }
