"""Action agent: side effects (file writes), always behind human approval.

    START -> propose -> approve (interrupt: wait for the user) -> execute -> END

`approve` calls LangGraph's `interrupt()`. The run stops there and is saved by
the checkpointer; the UI shows the proposed writes; the API resumes the run with
the user's decision (`Command(resume={"approved": ..., "note": ...})`). Only
then does `execute` touch the workspace. Nothing is written without a "yes".

The model never has to retype long content: an action can say "save the output
of step N", which code copies exactly.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, model_validator

from orchestraos.agents.base import Agent, AgentState
from orchestraos.agents.common import AgentError
from orchestraos.agents.prompts import ACTION_SYSTEM
from orchestraos.graph.answer import drop_save_instructions, strip_outer_fence
from orchestraos.llm.router import ModelRouter, ModelRouterError, TaskKind
from orchestraos.tools.client import ToolCaller

PREVIEW_CHARS = 800
MAX_ACTIONS = 5


class ProposedWrite(BaseModel):
    path: str = Field(min_length=1)
    from_step: int | None = None  # copy that earlier step's output
    content: str | None = None  # or literal content

    @model_validator(mode="after")
    def one_source(self) -> "ProposedWrite":
        if (self.from_step is None) == (self.content is None):
            raise ValueError("give exactly one of from_step or content")
        return self


class Proposal(BaseModel):
    actions: list[ProposedWrite] = Field(min_length=1, max_length=MAX_ACTIONS)


class ActionState(AgentState, total=False):
    writes: list[dict[str, Any]]
    decision: dict[str, Any]


MIN_COPIED_CHARS = 40


def copied_from(content: str, results: list[dict]) -> dict | None:
    """The earlier step whose output `content` is a (truncated) copy of, if any.

    Small models asked to "save the report" sometimes retype the start of it as
    literal content instead of using from_step, which saves a cut-off file.
    """
    snippet = content.strip().rstrip(".…").strip()
    if len(snippet) < MIN_COPIED_CHARS:
        return None
    for result in reversed(results):
        if result["ok"] and snippet in strip_outer_fence(result["output"]):
            return result
    return None


def resolve_writes(proposal: Proposal, results: list[dict], existing: set[str]) -> list[dict]:
    """Turn the model's proposal into concrete writes with the exact content to save."""
    by_step = {r["step_id"]: r for r in results}
    writes = []
    for action in proposal.actions:
        if action.from_step is not None:
            source = by_step.get(action.from_step)
            if source is None or not source["ok"]:
                raise AgentError(f"step {action.from_step} has no output to save")
            content = strip_outer_fence(source["output"])
        elif source := copied_from(action.content, results):
            content = strip_outer_fence(source["output"])  # save all of it, not the fragment
        else:
            content = action.content
        path = action.path.strip().lstrip("/").replace("\\", "/")
        content = drop_save_instructions(content, [path])  # no "save this to <path>" in the file
        writes.append({"path": path, "content": content, "overwrite": path in existing})
    return writes


def approval_request(writes: list[dict]) -> dict:
    """What the UI shows the user: every write, with size and a preview."""
    return {
        "kind": "approval",
        "actions": [
            {
                "tool": "write_file",
                "path": w["path"],
                "overwrite": w["overwrite"],
                "chars": len(w["content"]),
                "preview": w["content"][:PREVIEW_CHARS],
            }
            for w in writes
        ],
    }


def build_action_agent(router: ModelRouter, tools: ToolCaller) -> Agent:
    async def propose(state: ActionState) -> ActionState:
        # Only a short label per step: showing more invites the model to retype it.
        steps = "\n".join(
            f"- step {r['step_id']}: {r['agent']} output"
            + (" (FAILED, nothing to save)" if not r["ok"] else f", {len(r['output'])} chars")
            for r in state.get("results", [])
        )
        prompt = f"Instruction: {state['instruction']}\n\nEarlier steps:\n{steps or '(none)'}"
        messages = [SystemMessage(ACTION_SYSTEM), HumanMessage(prompt)]
        try:
            result = await router.ainvoke_json(TaskKind.PLAN, messages, Proposal)
        except ModelRouterError as exc:
            raise AgentError(f"couldn't work out which files to write: {exc}") from exc
        existing = set(await tools.call("list_files"))
        return {"writes": resolve_writes(result.parsed, state.get("results", []), existing)}

    def approve(state: ActionState) -> ActionState:
        # Pauses the whole run here until the API resumes it with the user's decision.
        decision = interrupt(approval_request(state["writes"]))
        return {"decision": {"approved": bool(decision.get("approved")),
                             "note": str(decision.get("note", ""))}}  # fmt: skip

    async def execute(state: ActionState) -> ActionState:
        paths = [w["path"] for w in state["writes"]]
        summary = "write_file " + ", ".join(paths)
        approval = {"action": summary, **state["decision"]}
        if not state["decision"]["approved"]:
            note = f" ({state['decision']['note']})" if state["decision"]["note"] else ""
            return {
                "output": f"Not saved: you declined writing {', '.join(paths)}{note}.",
                "sources": [], "artifacts": [], "approval": approval,
            }  # fmt: skip
        written = [
            await tools.call("write_file", path=w["path"], content=w["content"])
            for w in state["writes"]
        ]
        return {
            "output": f"Saved to {', '.join(written)} (approved by you).",
            "sources": [], "artifacts": written, "approval": approval,
        }  # fmt: skip

    graph = StateGraph(ActionState, input_schema=AgentState, output_schema=AgentState)
    graph.add_node("propose", propose)
    graph.add_node("approve", approve)
    graph.add_node("execute", execute)
    graph.add_edge(START, "propose")
    graph.add_edge("propose", "approve")
    graph.add_edge("approve", "execute")
    graph.add_edge("execute", END)
    return graph.compile(name="action_agent")
