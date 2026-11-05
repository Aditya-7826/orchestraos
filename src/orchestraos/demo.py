"""`orchestraos demo`: a guided, step-by-step tour of the real system in the terminal.

    uv run orchestraos demo              # pauses between steps, asks you to approve the write
    uv run orchestraos demo --yes        # no pauses, approves automatically
    uv run orchestraos demo --step 3     # just one step (1-5)

Each step says what to look for, runs a real goal through the full system (local models,
MCP tools, Docker sandbox, memory) and shows what happened: the triage decision, the plan,
each agent's result, the answer and its sources/files.

The tour keeps its memory in data/demo/, so it doesn't add preferences to your real memory.
Files it creates (a chart, a saved note) go in your workspace/ folder, where you can open them.
"""

import asyncio
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from orchestraos.api.events import to_events
from orchestraos.config import Settings
from orchestraos.doctor import format_check, run_checks
from orchestraos.graph.factory import open_app

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
RULE = "-" * 72
MEMORY_LINES_SHOWN = 8


@dataclass(frozen=True)
class Step:
    title: str
    explain: str  # what's about to happen, shown before the step runs
    goals: list[str] = field(default_factory=list)
    look_for: str = ""  # what to notice, shown after


STEPS = [
    Step(
        title="Check the setup",
        explain="The same checks as `make doctor`: Ollama is running, the three models are "
        "pulled, Docker works and the sandbox image is built.",
    ),
    Step(
        title="Ask a simple question",
        explain="The small model (triage) decides this needs no tools, so the plan is a "
        "single 'writing' step answered by the large model.",
        goals=["In two sentences, what does a multi-agent AI system do?"],
        look_for="Triage chose 'direct', and the plan has one writing step. There are no "
        "sources because nothing was researched (made-up citations are removed by code).",
    ),
    Step(
        title="Analyse a data file",
        explain="The planner routes this to the analytics agent. The agent profiles sales.csv, "
        "writes pandas/matplotlib code and runs it in a Docker sandbox with no network. "
        "Then the writer turns the printed numbers into the answer. This takes about 2 min.",
        goals=["Analyse sales.csv: total revenue per region, with a bar chart"],
        look_for="Expect North 106,080 and South 79,920 (exact, computed from the CSV), "
        "plus a chart file you can open.",
    ),
    Step(
        title="Approve a side effect",
        explain="Saving a file is a side effect, so only the action agent can do it. It "
        "pauses the run with LangGraph's interrupt() and asks you first.",
        goals=[
            "Write a 3-item checklist for reviewing a pull request "
            "and save it to notes/demo-checklist.md"
        ],  # fmt: skip
        look_for="Nothing was written until you answered. If you rejected it, the file "
        "doesn't exist and the answer says 'Not saved'.",
    ),
    Step(
        title="Memory across sessions",
        explain="First you state a preference. Then, in a new session, you ask something "
        "unrelated. The memory agent recalls the preference before planning.",
        goals=["I prefer answers as short bullet points. What is Docker?", "What is Kubernetes?"],
        look_for="The second run's 'Recalled from memory' shows your preference, and the "
        "answer uses short bullet points.",
    ),
]


class Console:
    """Terminal I/O, injectable so tests can script the answers."""

    def __init__(self, auto: bool, ask: Callable[[str], str] = input, show=print):
        self.auto, self._ask, self.show = auto, ask, show

    def pause(self) -> None:
        if not self.auto:
            self._ask("\nPress Enter to run this step... ")

    def confirm(self, question: str) -> bool:
        if self.auto:
            self.show(f"{question} y  (--yes)")
            return True
        return self._ask(f"{question} ").strip().lower() in {"y", "yes"}


def describe(event: dict) -> list[str]:
    """One run event as indented lines of plain text."""
    if event["type"] == "triage":
        decision = "direct answer" if event["mode"] == "direct" else "needs a plan"
        return [f"  triage (small model): {decision}"]
    if event["type"] == "plan":
        lines = ["  plan (large model):"]
        lines += [f"    {s['id']}. {s['agent']}: {s['instruction']}" for s in event["steps"]]
        return lines
    if event["type"] == "step":
        status = "ok" if event["ok"] else "FAILED"
        preview = " ".join(event["preview"].split())[:160]
        return [f"  step {event['step_id']} {event['agent']}: {status} - {preview}"]
    return []


async def run_goal(app: CompiledStateGraph, goal: str, console: Console) -> dict:
    """Run one goal, printing progress and asking about any approval. Returns the final state."""
    config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}
    console.show(f'\n> "{goal}"')
    graph_input: object = {"goal": goal}
    while True:
        async for chunk in app.astream(graph_input, config, stream_mode="updates"):
            for event in to_events((), "updates", chunk):
                for line in describe(event):
                    console.show(line)
        snapshot = await app.aget_state(config)
        pending = [i.value for t in snapshot.tasks for i in t.interrupts]
        if not pending:
            return snapshot.values
        console.show("\n  The run is paused. It wants to make these changes:")
        for action in pending[0]["actions"]:
            overwrite = " (OVERWRITES an existing file)" if action["overwrite"] else ""
            console.show(
                f"    {action['tool']} -> {action['path']}, {action['chars']} chars{overwrite}"
            )
            for line in action["preview"].splitlines()[:8]:
                console.show(f"      | {line}")
        approved = console.confirm("  Approve? [y/N]")
        graph_input = Command(resume={"approved": approved, "note": "from the demo"})


def show_result(state: dict, settings: Settings, console: Console) -> None:
    if memory := state.get("memory_context"):
        lines = memory.splitlines()
        console.show("\n  Recalled from memory:")
        console.show("\n".join(f"    {line}" for line in lines[:MEMORY_LINES_SHOWN]))
        if len(lines) > MEMORY_LINES_SHOWN:
            console.show(f"    ... ({len(lines) - MEMORY_LINES_SHOWN} more lines)")
    console.show("\n  Answer:")
    console.show("\n".join(f"    {line}" for line in state.get("final_answer", "").splitlines()))
    if sources := state.get("sources"):
        console.show("  Sources: " + ", ".join(sources))
    for artifact in state.get("artifacts", []):
        console.show(f"  File: {(settings.workspace_dir / artifact).resolve()}")


def demo_settings(settings: Settings) -> Settings:
    """Same models and workspace, but separate memory/checkpoints and no Neo4j or tracing."""
    return settings.model_copy(
        update={"data_dir": settings.data_dir / "demo", "neo4j_uri": None, "tracing": False}
    )


def ensure_sample_data(workspace: Path, console: Console) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / "sales.csv").exists() and (EXAMPLES / "sales.csv").exists():
        shutil.copy(EXAMPLES / "sales.csv", workspace / "sales.csv")
        console.show(f"(copied examples/sales.csv into {workspace})")


async def run_demo(settings: Settings, console: Console, only: list[int] | None = None) -> int:
    settings = demo_settings(settings)
    chosen = [(n, s) for n, s in enumerate(STEPS, 1) if not only or n in only]
    console.show("OrchestraOS guided tour: everything runs locally on your machine.")

    if any(n == 1 for n, _ in chosen):
        console.show(f"\n{RULE}\nStep 1/{len(STEPS)}: {STEPS[0].title}\n{RULE}\n{STEPS[0].explain}")
        checks = await asyncio.to_thread(run_checks, settings)
        for check in checks:
            console.show(format_check(check))
        if any(not c.ok and not c.warn_only for c in checks):
            console.show("\nFix the failed checks above, then run the demo again.")
            return 1

    ensure_sample_data(settings.workspace_dir, console)
    async with open_app(settings) as app:
        for n, step in chosen:
            if not step.goals:
                continue
            console.show(f"\n{RULE}\nStep {n}/{len(STEPS)}: {step.title}\n{RULE}\n{step.explain}")
            console.pause()
            for goal in step.goals:
                show_result(await run_goal(app, goal, console), settings, console)
            if step.look_for:
                console.show(f"\n  What to notice: {step.look_for}")

    console.show(f"\n{RULE}\nThat's the tour. Next: `make up` and open http://localhost:8501 "
                 "to use the chat UI.")  # fmt: skip
    return 0
