"""The eval harness itself: answer key, checks, and the runner's approval handling."""

import csv
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fakes import fake_router, system_prompt

from evals import run as runner
from evals.checks import CHECKS, Outcome, evaluate, numbers_in

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
TASKS = {t["id"]: t for t in runner.load_tasks()}


def rows(name: str) -> list[dict]:
    with (EXAMPLES / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_numbers(task_id: str) -> set[float]:
    return {float(c["number"]) for c in TASKS[task_id]["checks"] if "number" in c}


# --- the answer key matches the data ----------------------------------------


def test_sales_answers_match_the_csv():
    sales = rows("sales.csv")
    revenue, units, by_month = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in sales:
        revenue[r["region"]] += int(r["revenue"])
        units[r["region"]] += int(r["units"])
        by_month[r["month"]] += int(r["revenue"])
    assert expected_numbers("a-revenue-region") == {float(v) for v in revenue.values()}
    assert expected_numbers("a-units-chart") == {float(v) for v in units.values()}
    best_month = max(by_month, key=by_month.get)
    assert best_month == "2026-06" and expected_numbers("a-best-month") == {by_month[best_month]}
    per_unit = sum(int(r["revenue"]) for r in sales) / sum(int(r["units"]) for r in sales)
    assert expected_numbers("a-price-per-unit") == {per_unit}


def test_employee_answers_match_the_csv():
    staff = rows("employees.csv")
    salaries = defaultdict(list)
    for r in staff:
        salaries[r["department"]].append(int(r["salary"]))
    averages = {round(sum(v) / len(v)) for v in salaries.values()}
    assert expected_numbers("a-avg-salary") == {float(a) for a in averages}
    assert expected_numbers("a-payroll") == {float(sum(int(r["salary"]) for r in staff))}
    top = max(staff, key=lambda r: int(r["salary"]))["employee"]
    assert {"contains": top} in TASKS["a-payroll"]["checks"]
    early = {r["employee"] for r in staff if int(r["start_year"]) < 2021}
    (listed,) = [
        c["contains_all"] for c in TASKS["a-early-starters"]["checks"] if "contains_all" in c
    ]
    assert set(listed) <= early


def test_benchmark_has_30_plus_tasks_with_known_checks():
    assert len(TASKS) >= 30
    for task in TASKS.values():
        assert task["kind"] in {"triage", "plan", "run"}
        for spec in task["checks"]:
            assert next(iter(spec)) in CHECKS, (task["id"], spec)


# --- checks -----------------------------------------------------------------


def test_numbers_ignore_commas_and_find_decimals():
    found = numbers_in("North: 106,080 and South 79,920.50 (2026-06)")
    assert found == [106080, 79920.5, 2026, 6]  # a date's "-06" isn't a negative number


@pytest.mark.parametrize(
    ("answer", "spec", "passed"),
    [
        ("Total: 106,080", {"number": 106080}, True),
        ("Total: 106,079", {"number": 106080}, False),
        ("about 52,333.33", {"number": 52333, "tolerance": 1}, True),
        ("Ben earns most", {"contains": "ben"}, True),
        ("Ben, Dara", {"contains_all": ["Ben", "Dara", "Gus"]}, False),
        ("> **Some steps failed.**", {"failure_notice": True}, True),
    ],
)
def test_answer_checks(answer, spec, passed):
    assert evaluate(spec, Outcome(answer=answer)).passed is passed


def test_invariants_catch_made_up_urls_and_images():
    outcome = Outcome(
        answer="See https://real.example/a and https://fake.example/b ![c](outputs/x/c.png)",
        sources=["https://real.example/a"],
        artifacts=[],
    )
    urls = evaluate({"no_unbacked_urls": True}, outcome)
    assert not urls.passed and "fake.example" in urls.detail
    assert not evaluate({"images_backed": True}, outcome).passed


def test_file_checks(tmp_path):
    (tmp_path / "ws" / "notes").mkdir(parents=True)
    (tmp_path / "ws" / "notes" / "a.md").write_text("x" * 150)
    outcome = Outcome(workspace=tmp_path / "ws", setup_files={"notes/a.md": "x" * 150})
    assert evaluate({"file_exists": "notes/a.md", "min_chars": 100}, outcome).passed
    assert not evaluate({"file_exists": "notes/a.md", "min_chars": 200}, outcome).passed
    assert evaluate({"file_absent": "poems/rain.txt"}, outcome).passed
    assert evaluate({"file_unchanged": "notes/a.md"}, outcome).passed
    assert evaluate({"outside_absent": "outside.md"}, outcome).passed


def test_unknown_check_is_an_error():
    with pytest.raises(KeyError):
        evaluate({"vibes": "good"}, Outcome())


# --- runner -----------------------------------------------------------------


def llm(messages) -> str:
    prompt = system_prompt(messages)
    if "decide how to handle" in prompt:
        return json.dumps({"mode": "direct"})
    if "planner" in prompt:
        return json.dumps({"steps": [{"agent": "writing", "instruction": "w"}]})
    return json.dumps({"actions": [{"path": "notes/review.md", "from_step": 1}]})


async def test_triage_and_plan_tasks_run_with_the_router(settings):
    router = fake_router(llm)
    triage = await runner.run_task(TASKS["t-hello"], settings, router)
    assert triage["passed"] and triage["mode"] == "direct"
    plan = await runner.run_task(TASKS["p-save-analysis"], settings, router)
    # The fake plan has no analytics step, so this must fail, and say why.
    assert not plan["passed"]
    assert plan["agents"] == ["writing", "action"]  # save step added for "save ... to x.md"


async def test_run_tasks_answer_approval_prompts(settings, monkeypatch):
    from test_action import supervisor, workspace_tools

    tools, written = workspace_tools()

    @asynccontextmanager
    async def fake_open_app(isolated):
        yield supervisor(llm, tools)

    monkeypatch.setattr(runner, "open_app", fake_open_app)
    task = {"id": "x", "category": "safety", "kind": "run", "approve": True,
            "goals": ["write it and save to notes/review.md"],
            "checks": [{"interrupted": True}]}  # fmt: skip
    result = await runner.run_task(task, settings, fake_router(llm))
    assert result["passed"], result["checks"]
    # Triaged "direct", so the writer answered the goal; the approved action saved it exactly.
    assert written == {"notes/review.md": "[writing stub] write it and save to notes/review.md"}


async def test_a_crashing_task_fails_with_the_reason(settings, monkeypatch):
    @asynccontextmanager
    async def broken_open_app(isolated):
        raise ConnectionError("ollama down")
        yield

    monkeypatch.setattr(runner, "open_app", broken_open_app)
    task = {"id": "x", "category": "c", "kind": "run", "goals": ["g"], "checks": []}
    result = await runner.run_task(task, settings, fake_router(llm))
    assert not result["passed"] and "ollama down" in result["error"]


def test_summary_table():
    results = [
        {"id": "a", "category": "triage", "passed": True, "seconds": 2.0, "checks": []},
        {"id": "b", "category": "triage", "passed": False, "seconds": 4.0,
         "checks": [{"name": "triage = plan", "passed": False, "detail": "got direct"}]},
    ]  # fmt: skip
    meta = {"started": "now", "small_model": "s", "large_model": "l", "embed_model": "e",
            "machine": "m", "commit": "c", "wall_minutes": 1}  # fmt: skip
    text = runner.summarize(results, meta)
    assert "| triage | 1/2 | 50% | 3 s |" in text
    assert "- `b`: triage = plan — got direct" in text
