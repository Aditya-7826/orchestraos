import json

import pytest
from fakes import fake_router, system_prompt

from orchestraos.agents.base import AgentState, build_function_agent, build_stub_agent
from orchestraos.graph.checkpoint import open_checkpointer
from orchestraos.graph.planner import MAX_STEPS, Plan, PlanStep, normalize_plan
from orchestraos.graph.state import AGENT_NAMES
from orchestraos.graph.supervisor import build_supervisor, format_results
from orchestraos.memory.base import NullMemory


def responder(mode: str = "plan", plan: list[dict] | None = None, raw_plan: str | None = None):
    """Fake LLM: answers triage with `mode` and planning with `plan`."""

    def respond(messages) -> str:
        prompt = system_prompt(messages)
        if "decide how to handle" in prompt:
            return json.dumps({"mode": mode})
        if "planner" in prompt:
            return raw_plan if raw_plan is not None else json.dumps({"steps": plan or []})
        raise AssertionError(f"unexpected prompt: {prompt[:60]}")

    return respond


def stub_agents(**overrides):
    agents = {name: build_stub_agent(name) for name in AGENT_NAMES}
    agents.update(overrides)
    return agents


async def run(app, goal: str, thread: str = "t1") -> dict:
    return await app.ainvoke({"goal": goal}, {"configurable": {"thread_id": thread}})


class RecordingMemory(NullMemory):
    def __init__(self, recalled: str = ""):
        self.recalled = recalled
        self.stored: list[tuple[str, str]] = []

    async def recall(self, goal: str) -> str:
        return self.recalled

    async def remember(self, goal: str, answer: str) -> None:
        self.stored.append((goal, answer))


# --- end-to-end flows -------------------------------------------------------


async def test_direct_goal_goes_straight_to_writing():
    app = build_supervisor(fake_router(responder(mode="direct")), stub_agents(), NullMemory())
    state = await app.ainvoke({"goal": "What is LangGraph?"})
    assert [s["agent"] for s in state["plan"]] == ["writing"]
    assert state["final_answer"] == "[writing stub] What is LangGraph?"


async def test_planned_goal_runs_each_step_in_order_and_collects_sources():
    def research(task: AgentState) -> AgentState:
        return {"output": "found", "sources": ["https://a.example", "https://b.example"]}

    def writing(task: AgentState) -> AgentState:
        assert "found" in task["context"]  # earlier results are passed along
        return {"output": "the report", "sources": ["https://a.example"]}

    plan = [
        {"agent": "research", "instruction": "look it up"},
        {"agent": "analytics", "instruction": "crunch"},
        {"agent": "writing", "instruction": "write it"},
    ]
    agents = stub_agents(
        research=build_function_agent("research", research),
        writing=build_function_agent("writing", writing),
    )
    app = build_supervisor(fake_router(responder(plan=plan)), agents, NullMemory())

    state = await app.ainvoke({"goal": "compare things"})
    assert [r["agent"] for r in state["results"]] == ["research", "analytics", "writing"]
    assert state["final_answer"] == "the report"
    assert state["sources"] == ["https://a.example", "https://b.example"]


async def test_failing_agent_is_recorded_and_run_continues():
    def broken(task: AgentState) -> AgentState:
        raise RuntimeError("search is down")

    plan = [{"agent": "research", "instruction": "x"}, {"agent": "writing", "instruction": "y"}]
    agents = stub_agents(research=build_function_agent("research", broken))
    app = build_supervisor(fake_router(responder(plan=plan)), agents, NullMemory())

    state = await app.ainvoke({"goal": "g"})
    research_result = state["results"][0]
    assert not research_result["ok"] and "search is down" in research_result["output"]
    assert state["final_answer"] == "[writing stub] y"


async def test_unparseable_plan_falls_back_to_single_writing_step():
    app = build_supervisor(
        fake_router(responder(raw_plan="I cannot plan this")), stub_agents(), NullMemory()
    )
    state = await app.ainvoke({"goal": "do the thing"})
    assert [s["agent"] for s in state["plan"]] == ["writing"]


async def test_unparseable_triage_defaults_to_planning():
    def respond(messages) -> str:
        if "decide how to handle" in system_prompt(messages):
            return "uhh"
        return json.dumps({"steps": [{"agent": "research", "instruction": "r"}]})

    app = build_supervisor(fake_router(respond), stub_agents(), NullMemory())
    state = await app.ainvoke({"goal": "g"})
    assert state["mode"] == "plan"
    assert [s["agent"] for s in state["plan"]] == ["research", "writing"]


async def test_memory_is_recalled_before_planning_and_stored_after():
    memory = RecordingMemory(recalled="user prefers short answers")
    seen_prompts: list[str] = []

    def respond(messages) -> str:
        seen_prompts.append(str(messages[-1].content))
        return json.dumps({"mode": "direct"})

    app = build_supervisor(fake_router(respond), stub_agents(), memory)
    state = await app.ainvoke({"goal": "hello"})
    assert "user prefers short answers" in seen_prompts[0]
    assert memory.stored == [("hello", state["final_answer"])]


def test_requires_every_agent():
    agents = stub_agents()
    del agents["analytics"]
    with pytest.raises(ValueError, match="analytics"):
        build_supervisor(fake_router(responder()), agents, NullMemory())


# --- checkpointing ----------------------------------------------------------


class SimulatedCrash(BaseException):
    """Stands in for the process being killed (not an Exception, so agents can't catch it)."""


async def test_run_resumes_from_checkpoint_after_restart(tmp_path):
    db = tmp_path / "checkpoints.sqlite"
    calls = {"research": 0, "analytics": 0}

    def research(task: AgentState) -> AgentState:
        calls["research"] += 1
        return {"output": "facts"}

    def analytics(task: AgentState) -> AgentState:
        calls["analytics"] += 1
        if calls["analytics"] == 1:
            raise SimulatedCrash
        return {"output": "numbers"}

    plan = [
        {"agent": "research", "instruction": "r"},
        {"agent": "analytics", "instruction": "a"},
        {"agent": "writing", "instruction": "w"},
    ]
    agents = stub_agents(
        research=build_function_agent("research", research),
        analytics=build_function_agent("analytics", analytics),
    )
    router = fake_router(responder(plan=plan))

    # First "process": crashes during the analytics step.
    async with open_checkpointer(db) as checkpointer:
        app = build_supervisor(router, agents, NullMemory(), checkpointer)
        with pytest.raises(SimulatedCrash):
            await run(app, "goal", thread="resume-me")

    # Second "process": new connection, new compiled graph, same thread id.
    async with open_checkpointer(db) as checkpointer:
        app = build_supervisor(router, agents, NullMemory(), checkpointer)
        state = await app.ainvoke(None, {"configurable": {"thread_id": "resume-me"}})

    assert [r["agent"] for r in state["results"]] == ["research", "analytics", "writing"]
    assert calls == {"research": 1, "analytics": 2}  # research was not re-run


async def test_threads_are_isolated(tmp_path):
    async with open_checkpointer(tmp_path / "c.sqlite") as checkpointer:
        app = build_supervisor(
            fake_router(responder(mode="direct")), stub_agents(), NullMemory(), checkpointer
        )
        await run(app, "first", thread="a")
        await run(app, "second", thread="b")
        snapshot = await app.aget_state({"configurable": {"thread_id": "a"}})
    assert snapshot.values["goal"] == "first"


# --- helpers ----------------------------------------------------------------


def test_normalize_plan_caps_steps_and_ends_with_writing():
    plan = Plan(steps=[PlanStep(agent="research", instruction=f"r{i}") for i in range(10)])
    steps = normalize_plan(plan)
    assert len(steps) == MAX_STEPS
    assert steps[-1]["agent"] == "writing"
    assert [s["id"] for s in steps] == list(range(1, MAX_STEPS + 1))


def test_normalize_plan_keeps_valid_plan():
    plan = Plan(steps=[PlanStep(agent="writing", instruction="w")])
    assert normalize_plan(plan) == [{"id": 1, "agent": "writing", "instruction": "w"}]


def test_format_results_marks_failures_and_lists_sources():
    text = format_results(
        [
            {"step_id": 1, "agent": "research", "ok": True, "output": "o",
             "sources": ["https://x"], "artifacts": []},
            {"step_id": 2, "agent": "analytics", "ok": False, "output": "Error: boom",
             "sources": [], "artifacts": ["chart.png"]},
        ]
    )  # fmt: skip
    assert "Sources: https://x" in text
    assert "analytics (FAILED)" in text
    assert "Files: chart.png" in text
