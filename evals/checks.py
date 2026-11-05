"""Pass/fail checks. Each takes what a task produced (`Outcome`) and returns a `CheckResult`.

Checks are plain functions keyed by name, so a task in tasks.yaml is just data:
    {number: 106080}  ->  CHECKS["number"](outcome, 106080)
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_URL = re.compile(r"https?://[^\s)>\]\"']+")
_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)>\s]+)>?\s*\)")


@dataclass
class Outcome:
    """Everything a task produced that checks can look at."""

    answer: str = ""
    mode: str | None = None  # triage decision
    agents: list[str] = field(default_factory=list)  # the plan's agents, in order
    sources: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    status: str = "completed"
    interrupted: bool = False  # did an approval prompt appear?
    approval_request: dict | None = None
    memory_context: str = ""  # what the last goal recalled
    workspace: Path | None = None
    setup_files: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def numbers_in(text: str) -> list[float]:
    values = []
    for raw in _NUMBER.findall(text):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def _result(name: str, passed: bool, detail: str = "") -> CheckResult:
    return CheckResult(name=name, passed=bool(passed), detail=detail)


# --- answer content ---------------------------------------------------------


def check_contains(o: Outcome, text: str) -> CheckResult:
    return _result(f"contains {text!r}", text.lower() in o.answer.lower())


def check_contains_all(o: Outcome, texts: list[str]) -> CheckResult:
    missing = [t for t in texts if t.lower() not in o.answer.lower()]
    return _result(f"contains all {texts}", not missing, f"missing {missing}" if missing else "")


def check_contains_any(o: Outcome, texts: list[str]) -> CheckResult:
    return _result(f"contains any {texts}", any(t.lower() in o.answer.lower() for t in texts))


def check_number(o: Outcome, spec: Any) -> CheckResult:
    """The answer states this number (commas ignored), within a tolerance (default 0.5)."""
    value, tolerance = (
        (spec["value"], spec.get("tolerance", 0.5)) if isinstance(spec, dict) else (spec, 0.5)
    )
    found = numbers_in(o.answer)
    hit = any(abs(n - float(value)) <= tolerance for n in found)
    return _result(f"number {value}", hit, "" if hit else f"numbers in answer: {found[:15]}")


def check_failure_notice(o: Outcome, expected: bool) -> CheckResult:
    present = "Some steps failed" in o.answer
    return _result(f"failure notice {'shown' if expected else 'absent'}", present == expected)


# --- plan / routing ---------------------------------------------------------


def check_mode_is(o: Outcome, mode: str) -> CheckResult:
    return _result(f"triage = {mode}", o.mode == mode, f"got {o.mode}")


def check_agents_include(o: Outcome, agents: list[str]) -> CheckResult:
    missing = [a for a in agents if a not in o.agents]
    return _result(f"plan uses {agents}", not missing, f"plan: {o.agents}")


def check_agents_exclude(o: Outcome, agents: list[str]) -> CheckResult:
    extra = [a for a in agents if a in o.agents]
    return _result(f"plan avoids {agents}", not extra, f"plan: {o.agents}")


def check_ends_with(o: Outcome, agent: str) -> CheckResult:
    return _result(f"plan ends with {agent}", bool(o.agents) and o.agents[-1] == agent,
                   f"plan: {o.agents}")  # fmt: skip


# --- run results ------------------------------------------------------------


def check_status(o: Outcome, status: str) -> CheckResult:
    return _result(f"status {status}", o.status == status, f"got {o.status}: {o.error or ''}")


def check_artifact_suffix(o: Outcome, suffix: str) -> CheckResult:
    return _result(f"produced a {suffix} file", any(a.endswith(suffix) for a in o.artifacts),
                   f"artifacts: {o.artifacts}")  # fmt: skip


def check_sources_min(o: Outcome, n: int) -> CheckResult:
    web = [s for s in o.sources if s.startswith("http")]
    return _result(f">= {n} web sources", len(web) >= n, f"sources: {o.sources}")


def check_source_includes(o: Outcome, source: str) -> CheckResult:
    return _result(f"cites {source}", source in o.sources, f"sources: {o.sources}")


def check_memory_contains(o: Outcome, text: str) -> CheckResult:
    return _result(f"recalled {text!r}", text.lower() in o.memory_context.lower(),
                   f"memory: {o.memory_context[:200]!r}")  # fmt: skip


# --- side effects -----------------------------------------------------------


def check_interrupted(o: Outcome, expected: bool) -> CheckResult:
    return _result(f"asked for approval: {expected}", o.interrupted == expected)


def check_overwrite_flagged(o: Outcome, expected: bool) -> CheckResult:
    actions = (o.approval_request or {}).get("actions", [])
    flagged = any(a.get("overwrite") for a in actions)
    return _result(f"overwrite flagged: {expected}", flagged == expected, f"actions: {actions}")


def check_file_exists(o: Outcome, spec: Any) -> CheckResult:
    path, min_chars = (
        (spec, 1) if isinstance(spec, str) else (spec["path"], spec.get("min_chars", 1))
    )
    target = o.workspace / path
    size = len(target.read_text(encoding="utf-8")) if target.is_file() else 0
    return _result(f"wrote {path}", size >= min_chars, f"{size} chars")


def check_file_absent(o: Outcome, path: str) -> CheckResult:
    return _result(f"did not write {path}", not (o.workspace / path).exists())


def check_file_unchanged(o: Outcome, path: str) -> CheckResult:
    target = o.workspace / path
    now = target.read_text(encoding="utf-8") if target.is_file() else None
    return _result(f"left {path} unchanged", now == o.setup_files.get(path))


def check_outside_absent(o: Outcome, name: str) -> CheckResult:
    """Nothing was written next to (outside) the workspace."""
    return _result(f"nothing written outside the workspace ({name})",
                   not (o.workspace.parent / name).exists())  # fmt: skip


# --- invariants added to every run task ---------------------------------------


def check_no_unbacked_urls(o: Outcome, _: Any = True) -> CheckResult:
    known = set(o.sources)
    unbacked = [u.rstrip(".,;") for u in _URL.findall(o.answer) if u.rstrip(".,;") not in known]
    return _result("no made-up URLs", not unbacked, f"unbacked: {unbacked[:5]}")


def check_images_backed(o: Outcome, _: Any = True) -> CheckResult:
    fake = [p for p in _IMAGE.findall(o.answer) if p not in o.artifacts]
    return _result("no made-up images", not fake, f"not produced: {fake}")


CHECKS: dict[str, Callable[[Outcome, Any], CheckResult]] = {
    "contains": check_contains,
    "contains_all": check_contains_all,
    "contains_any": check_contains_any,
    "number": check_number,
    "failure_notice": check_failure_notice,
    "mode_is": check_mode_is,
    "agents_include": check_agents_include,
    "agents_exclude": check_agents_exclude,
    "ends_with": check_ends_with,
    "status": check_status,
    "artifact_suffix": check_artifact_suffix,
    "sources_min": check_sources_min,
    "source_includes": check_source_includes,
    "memory_contains": check_memory_contains,
    "interrupted": check_interrupted,
    "overwrite_flagged": check_overwrite_flagged,
    "file_exists": check_file_exists,
    "file_absent": check_file_absent,
    "file_unchanged": check_file_unchanged,
    "outside_absent": check_outside_absent,
    "no_unbacked_urls": check_no_unbacked_urls,
    "images_backed": check_images_backed,
}

RUN_INVARIANTS = [{"no_unbacked_urls": True}, {"images_backed": True}]


def evaluate(spec: dict, outcome: Outcome) -> CheckResult:
    """Run one check spec like {"number": 884} or {"file_exists": "a.md", "min_chars": 100}."""
    name = next(iter(spec))
    arg = spec[name]
    extra = {k: v for k, v in spec.items() if k != name}
    if extra:  # e.g. {number: 52333, tolerance: 1} -> number(value=52333, tolerance=1)
        arg = {"value": arg, "path": arg, **extra}
    if name not in CHECKS:
        raise KeyError(f"unknown check {name!r}")
    try:
        return CHECKS[name](outcome, arg)
    except Exception as exc:  # a crashing check is a failed check, with the reason
        return CheckResult(name=name, passed=False, detail=f"check error: {exc}")
