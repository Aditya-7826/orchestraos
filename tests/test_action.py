import json

import pytest
from fakes import FakeTools, fake_router, system_prompt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from orchestraos.agents.action import Proposal, ProposedWrite, build_action_agent, resolve_writes
from orchestraos.agents.base import build_stub_agent
from orchestraos.agents.common import AgentError
from orchestraos.agents.prompts import ACTION_SYSTEM
from orchestraos.graph.planner import (
    Plan,
    PlanStep,
    ensure_save_step,
    normalize_plan,
    requested_file,
)
from orchestraos.graph.state import AGENT_NAMES
from orchestraos.graph.supervisor import build_supervisor
from orchestraos.memory.base import NullMemory
from orchestraos.tools.client import restrict

PLAN = [
    {"agent": "writing", "instruction": "write the report"},
    {"agent": "action", "instruction": "save the report to reports/r.md"},
]


def llm(proposal: dict, counter: dict | None = None):
    def respond(messages) -> str:
        prompt = system_prompt(messages)
        if "decide how to handle" in prompt:
            return json.dumps({"mode": "plan"})
        if "planner" in prompt:
            return json.dumps({"steps": PLAN})
        assert prompt == ACTION_SYSTEM
        if counter is not None:
            counter["propose"] = counter.get("propose", 0) + 1
        return json.dumps(proposal)

    return respond


def workspace_tools(existing=()):
    written = {}

    def write_file(path, content):
        written[path] = content
        return path

    tools = FakeTools(list_files=lambda **_: list(existing), write_file=write_file)
    return tools, written


def supervisor(respond, tools):
    agents = {name: build_stub_agent(name) for name in AGENT_NAMES}
    agents["action"] = build_action_agent(fake_router(respond), tools)
    return build_supervisor(fake_router(respond), agents, NullMemory(), InMemorySaver())


CONFIG = {"configurable": {"thread_id": "t"}}
SAVE_STEP_1 = {"actions": [{"path": "reports/r.md", "from_step": 1}]}


async def test_run_pauses_for_approval_before_writing():
    tools, written = workspace_tools()
    app = supervisor(llm(SAVE_STEP_1), tools)
    await app.ainvoke({"goal": "report and save it"}, CONFIG)

    snapshot = await app.aget_state(CONFIG)
    assert snapshot.next == ("action",)
    (request,) = [i.value for t in snapshot.tasks for i in t.interrupts]
    assert request["kind"] == "approval"
    assert request["actions"][0]["path"] == "reports/r.md"
    assert request["actions"][0]["preview"] == "[writing stub] write the report"
    assert written == {}  # nothing happens before approval


async def test_approved_write_saves_the_exact_earlier_output():
    tools, written = workspace_tools()
    counter = {}
    app = supervisor(llm(SAVE_STEP_1, counter), tools)
    await app.ainvoke({"goal": "report and save it"}, CONFIG)
    state = await app.ainvoke(Command(resume={"approved": True, "note": ""}), CONFIG)

    assert written == {"reports/r.md": "[writing stub] write the report"}
    assert counter["propose"] == 1  # resuming doesn't re-ask the model
    assert state["approvals"] == [
        {"step_id": 2, "action": "write_file reports/r.md", "approved": True, "note": ""}
    ]
    assert state["final_answer"].endswith("**Saved to reports/r.md (approved by you).**")
    assert state["artifacts"] == ["reports/r.md"]


async def test_rejected_write_does_nothing():
    tools, written = workspace_tools()
    app = supervisor(llm(SAVE_STEP_1), tools)
    await app.ainvoke({"goal": "report and save it"}, CONFIG)
    state = await app.ainvoke(Command(resume={"approved": False, "note": "wrong folder"}), CONFIG)

    assert written == {}
    assert state["approvals"][0]["approved"] is False
    assert "Not saved: you declined writing reports/r.md (wrong folder)." in state["final_answer"]


async def test_overwrite_is_flagged_in_the_request():
    tools, _ = workspace_tools(existing=["reports/r.md"])
    app = supervisor(llm(SAVE_STEP_1), tools)
    await app.ainvoke({"goal": "g"}, CONFIG)
    snapshot = await app.aget_state(CONFIG)
    assert snapshot.tasks[0].interrupts[0].value["actions"][0]["overwrite"] is True


async def test_bad_proposal_fails_the_step_without_writing():
    tools, written = workspace_tools()
    app = supervisor(llm({"actions": []}), tools)
    state = await app.ainvoke({"goal": "g"}, CONFIG)
    action = state["results"][-1]
    assert action["agent"] == "action" and not action["ok"]
    assert written == {} and not (await app.aget_state(CONFIG)).next


# --- pieces -----------------------------------------------------------------


RESULTS = [
    {"step_id": 1, "agent": "writing", "ok": True, "output": "```markdown\n# R\n```",
     "sources": [], "artifacts": []},
    {"step_id": 2, "agent": "research", "ok": False, "output": "Error", "sources": [],
     "artifacts": []},
]  # fmt: skip


def test_resolve_writes_copies_step_output_and_cleans_paths():
    proposal = Proposal(actions=[ProposedWrite(path="/reports\\r.md", from_step=1)])
    writes = resolve_writes(proposal, RESULTS, existing=set())
    assert writes == [{"path": "reports/r.md", "content": "# R", "overwrite": False}]


@pytest.mark.parametrize("step", [2, 9])
def test_resolve_writes_rejects_failed_or_missing_steps(step):
    proposal = Proposal(actions=[ProposedWrite(path="r.md", from_step=step)])
    with pytest.raises(AgentError):
        resolve_writes(proposal, RESULTS, existing=set())


@pytest.mark.parametrize("fields", [{}, {"from_step": 1, "content": "x"}])
def test_action_needs_exactly_one_content_source(fields):
    with pytest.raises(ValidationError):
        ProposedWrite(path="r.md", **fields)


async def test_restricted_tools_block_everything_else():
    tools = restrict(FakeTools(read_file=lambda **_: "x"), "list_files")
    with pytest.raises(PermissionError, match="read_file"):
        await tools.call("read_file", path="a")


def test_plan_may_end_with_writing_then_action():
    plan = Plan(steps=[PlanStep(agent="writing", instruction="w"),
                       PlanStep(agent="action", instruction="save")])  # fmt: skip
    assert [s["agent"] for s in normalize_plan(plan)] == ["writing", "action"]


def test_plan_ending_in_action_without_writing_gets_a_writer():
    plan = Plan(steps=[PlanStep(agent="action", instruction="save")])
    assert [s["agent"] for s in normalize_plan(plan)] == ["action", "writing"]


def test_content_copied_from_a_step_saves_the_whole_step():
    output = "- First point about checkpointers and why they matter.\n- Second point."
    full = {"step_id": 1, "agent": "writing", "ok": True, "sources": [], "artifacts": [],
            "output": output}  # fmt: skip
    fragment = "- First point about checkpointers and why they matt..."
    proposal = Proposal(actions=[ProposedWrite(path="n.md", content=fragment)])
    (write,) = resolve_writes(proposal, [full], existing=set())
    assert write["content"] == full["output"]


def test_short_new_content_is_kept_as_is():
    proposal = Proposal(actions=[ProposedWrite(path="todo.txt", content="- call Sam")])
    (write,) = resolve_writes(proposal, RESULTS, existing=set())
    assert write["content"] == "- call Sam"


@pytest.mark.parametrize(
    ("goal", "path"),
    [
        ("write a checklist and save it to notes/pr-checklist.md", "notes/pr-checklist.md"),
        ("Summarise this and write the result into `out/summary.txt`", "out/summary.txt"),
        ("Export the table as report.csv please", "report.csv"),
        ("What is a vector database?", None),
        ("Analyse sales.csv and chart revenue", None),  # names a file, but not as a save target
    ],
)
def test_requested_file(goal, path):
    assert requested_file(goal) == path


def test_missing_save_step_is_added():
    steps = [{"id": 1, "agent": "writing", "instruction": "write it and save to x.md"}]
    fixed = ensure_save_step(steps, "write a note and save it to notes/x.md")
    assert [s["agent"] for s in fixed] == ["writing", "action"]
    assert fixed[-1]["id"] == 2
    assert fixed[-1]["instruction"] == "Save the answer from the previous step to notes/x.md"


def test_existing_action_step_is_left_alone():
    steps = [{"id": 1, "agent": "writing", "instruction": "w"},
             {"id": 2, "agent": "action", "instruction": "save"}]  # fmt: skip
    assert ensure_save_step(steps, "save it to a.md") == steps


def test_saved_file_has_no_save_instructions():
    output = "# R\n\n- point\n\nSave the above content to `reports/r.md`."
    results = [{"step_id": 1, "agent": "writing", "ok": True, "output": output,
                "sources": [], "artifacts": []}]  # fmt: skip
    proposal = Proposal(actions=[ProposedWrite(path="reports/r.md", from_step=1)])
    (write,) = resolve_writes(proposal, results, existing=set())
    assert write["content"] == "# R\n\n- point"
