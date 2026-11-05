"""Run the benchmark and write measured results.

    uv run python -m evals.run                      # everything (~1 h on an 8 GB laptop)
    uv run python -m evals.run --only triage planning
    uv run python -m evals.run --only a-units-chart
    uv run python -m evals.run --resume evals/results/20261105T200000

Results go to evals/results/<timestamp>/: results.jsonl (one line per task, written as
soon as the task finishes, so a crash loses nothing and --resume skips finished tasks),
run.json (models, machine, git commit) and summary.md (the tables used in the README).
"""

import argparse
import asyncio
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import psutil
import yaml
from langgraph.types import Command

from evals.checks import RUN_INVARIANTS, CheckResult, Outcome, evaluate
from orchestraos.config import Settings
from orchestraos.graph import planner
from orchestraos.graph.factory import open_app
from orchestraos.llm.router import ModelRouter

ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = ROOT / "evals" / "tasks.yaml"
RESULTS_DIR = ROOT / "evals" / "results"
EXAMPLES = ROOT / "examples"
TASK_TIMEOUT_SECONDS = 1200


def load_tasks(path: Path = TASKS_FILE) -> list[dict]:
    tasks = yaml.safe_load(path.read_text(encoding="utf-8"))["tasks"]
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    return tasks


def select(tasks: list[dict], only: list[str] | None) -> list[dict]:
    if not only:
        return tasks
    return [t for t in tasks if t["id"] in only or t["category"] in only]


# --- running one task -------------------------------------------------------


async def run_triage(task: dict, router: ModelRouter) -> Outcome:
    return Outcome(mode=await planner.triage(router, task["goal"]))


async def run_plan(task: dict, router: ModelRouter) -> Outcome:
    steps = planner.ensure_save_step(await planner.make_plan(router, task["goal"]), task["goal"])
    return Outcome(agents=[s["agent"] for s in steps])


def prepare_workspace(base: Path, setup_files: dict[str, str]) -> Path:
    """A fresh copy of the example files, plus any task-specific files."""
    workspace = base / "workspace"
    shutil.copytree(EXAMPLES, workspace)
    for rel, content in setup_files.items():
        (workspace / rel).parent.mkdir(parents=True, exist_ok=True)
        (workspace / rel).write_text(content, encoding="utf-8")
    return workspace


async def run_full(task: dict, settings: Settings, base: Path) -> Outcome:
    """The whole system, in an isolated workspace and memory store, answering approvals."""
    setup_files = task.get("setup_files", {})
    workspace = prepare_workspace(base, setup_files)
    isolated = settings.model_copy(
        update={
            "workspace_dir": workspace,
            "data_dir": base / "data",
            "tracing": False,
            "neo4j_uri": None,  # semantic memory only: fewer moving parts in the benchmark
            "research_mode": "local",
        }
    )
    outcome = Outcome(workspace=workspace, setup_files=setup_files)
    async with open_app(isolated) as app:
        for goal in task["goals"]:  # several goals = several sessions sharing memory
            config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}
            await app.ainvoke({"goal": goal}, config)
            snapshot = await app.aget_state(config)
            while snapshot.next:
                pending = [i.value for t in snapshot.tasks for i in t.interrupts]
                if not pending:
                    outcome.status = "stopped"
                    break
                outcome.interrupted = True
                outcome.approval_request = pending[0]
                decision = {"approved": bool(task.get("approve", False)), "note": "eval"}
                await app.ainvoke(Command(resume=decision), config)
                snapshot = await app.aget_state(config)

        values = snapshot.values
        outcome.answer = values.get("final_answer", "") or ""
        outcome.sources = values.get("sources", [])
        outcome.artifacts = values.get("artifacts", [])
        outcome.agents = [s["agent"] for s in values.get("plan", [])]
        outcome.memory_context = values.get("memory_context", "")
    return outcome


async def run_task(task: dict, settings: Settings, router: ModelRouter) -> dict:
    started = time.perf_counter()
    base = Path(tempfile.mkdtemp(prefix=f"orchestraos-eval-{task['id']}-"))
    try:
        if task["kind"] == "triage":
            work = run_triage(task, router)
        elif task["kind"] == "plan":
            work = run_plan(task, router)
        else:
            work = run_full(task, settings, base)
        outcome = await asyncio.wait_for(work, timeout=TASK_TIMEOUT_SECONDS)
    except Exception as exc:  # a crash or timeout fails the task, with the reason
        outcome = Outcome(status="error", error=f"{type(exc).__name__}: {exc}")

    specs = list(task["checks"])
    if task["kind"] == "run" and outcome.status != "error":
        specs += RUN_INVARIANTS
    checks: list[CheckResult] = (
        [CheckResult("task ran", False, outcome.error)]
        if outcome.status == "error"
        else [evaluate(spec, outcome) for spec in specs]
    )
    shutil.rmtree(base, ignore_errors=True)
    return {
        "id": task["id"],
        "category": task["category"],
        "kind": task["kind"],
        "passed": all(c.passed for c in checks),
        "seconds": round(time.perf_counter() - started, 1),
        "checks": [vars(c) for c in checks],
        "agents": outcome.agents,
        "mode": outcome.mode,
        "answer": outcome.answer[:2000],
        "error": outcome.error,
    }


# --- reporting --------------------------------------------------------------


def summarize(results: list[dict], meta: dict) -> str:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    def row(name: str, rs: list[dict]) -> str:
        passed = sum(r["passed"] for r in rs)
        median = statistics.median(r["seconds"] for r in rs)
        return f"| {name} | {passed}/{len(rs)} | {100 * passed / len(rs):.0f}% | {median:.0f} s |"

    lines = [
        f"# Eval results — {meta['started']}",
        "",
        f"Models: small `{meta['small_model']}`, large `{meta['large_model']}`, "
        f"embeddings `{meta['embed_model']}` · machine: {meta['machine']} · "
        f"commit `{meta['commit']}` · wall time {meta.get('wall_minutes', '?')} min",
        "",
        "| Category | Passed | Rate | Median time |",
        "|---|---|---|---|",
        *(row(name, rs) for name, rs in sorted(by_category.items())),
        row("**all**", results),
        "",
    ]
    failures = [r for r in results if not r["passed"]]
    if failures:
        lines += ["## Failures", ""]
        for r in failures:
            first = next(c for c in r["checks"] if not c["passed"])
            lines.append(f"- `{r['id']}`: {first['name']} — {first['detail']}".rstrip(" —"))
    return "\n".join(lines) + "\n"


def preflight(settings: Settings) -> dict:
    """Warn about things that have wrecked long runs on this kind of machine before."""
    memory = psutil.virtual_memory()
    battery = psutil.sensors_battery()
    if battery is not None and not battery.power_plugged:
        print("WARNING: on battery: the GPU throttles and the laptop may sleep. Plug in.")
    if memory.available < 3 * 1024**3:
        print(f"WARNING: only {memory.available / 1024**3:.1f} GB RAM free; close other apps "
              "and stop optional Docker profiles (make down) first.")  # fmt: skip
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()  # fmt: skip
    return {
        "started": datetime.now().isoformat(timespec="seconds"),
        "small_model": settings.small_model,
        "large_model": settings.large_model,
        "embed_model": settings.embed_model,
        "machine": f"{platform.system()} {platform.release()}, "
        f"{memory.total / 1024**3:.1f} GB RAM, {psutil.cpu_count()} CPUs",
        "commit": commit or "unknown",
    }


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.run")
    parser.add_argument("--only", nargs="*", help="task ids or categories")
    parser.add_argument("--resume", type=Path, help="results folder to continue")
    args = parser.parse_args(argv)

    settings = Settings()
    tasks = select(load_tasks(), args.only)
    out_dir = args.resume or RESULTS_DIR / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.jsonl"
    done = (
        [json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines()]
        if results_file.exists()
        else []
    )
    done_ids = {r["id"] for r in done}
    meta_file = out_dir / "run.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else preflight(settings)
    meta_file.write_text(json.dumps(meta, indent=2))

    router = ModelRouter.from_settings(settings)
    wall = time.perf_counter()
    todo = [t for t in tasks if t["id"] not in done_ids]
    print(f"{len(todo)} task(s) to run, {len(done_ids)} already done -> {out_dir}")
    for i, task in enumerate(todo, 1):
        result = await run_task(task, settings, router)
        with results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        done.append(result)
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"[{i}/{len(todo)}] {mark} {task['id']} ({result['seconds']} s)", flush=True)

    meta["wall_minutes"] = round(meta.get("wall_minutes", 0) + (time.perf_counter() - wall) / 60)
    meta_file.write_text(json.dumps(meta, indent=2))
    summary = summarize([r for r in done if r["id"] in {t["id"] for t in tasks}], meta)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
