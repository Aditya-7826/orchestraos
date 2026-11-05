"""Triage (small model) and planning (large model).

Both use `ModelRouter.invoke_json`, so a bad answer is retried and falls back
to the other model. If even that fails we don't crash the run: triage defaults
to "plan" (the safer choice) and planning defaults to a single writing step.
"""

import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from orchestraos.graph.prompts import PLANNER_SYSTEM, TRIAGE_SYSTEM, with_memory
from orchestraos.graph.state import AgentName, Step
from orchestraos.llm.router import ModelRouter, ModelRouterError, TaskKind

logger = logging.getLogger(__name__)

MAX_STEPS = 6
FINAL_WRITING_INSTRUCTION = "Write the final answer for the user from the results so far."


class Triage(BaseModel):
    mode: Literal["direct", "plan"]


class PlanStep(BaseModel):
    agent: AgentName
    instruction: str = Field(min_length=1)


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1)


async def triage(
    router: ModelRouter, goal: str, memory_context: str = ""
) -> Literal["direct", "plan"]:
    messages = [SystemMessage(TRIAGE_SYSTEM), HumanMessage(with_memory(goal, memory_context))]
    try:
        result = await router.ainvoke_json(TaskKind.CLASSIFY, messages, Triage)
    except ModelRouterError:
        logger.warning("triage failed; defaulting to 'plan'")
        return "plan"
    return result.parsed.mode


def direct_plan(goal: str) -> list[Step]:
    return [Step(id=1, agent="writing", instruction=goal)]


async def make_plan(router: ModelRouter, goal: str, memory_context: str = "") -> list[Step]:
    messages = [
        SystemMessage(PLANNER_SYSTEM.format(max_steps=MAX_STEPS)),
        HumanMessage(with_memory(goal, memory_context)),
    ]
    try:
        result = await router.ainvoke_json(TaskKind.PLAN, messages, Plan)
    except ModelRouterError:
        logger.warning("planning failed; falling back to a single writing step")
        return direct_plan(goal)
    return normalize_plan(result.parsed)


_SAVE_REQUEST = re.compile(
    r"\b(?:save|write|export|store|put)\b[^.?!]*?\b(?:to|as|in|into)\s+[`'\"]?"
    r"([\w./\\-]+\.(?:md|txt|csv|json|html))",
    re.IGNORECASE,
)


def requested_file(goal: str) -> str | None:
    """The file the user asked to save to, e.g. 'notes/x.md' in '... and save it to notes/x.md'."""
    match = _SAVE_REQUEST.search(goal)
    return match.group(1) if match else None


def ensure_save_step(steps: list[Step], goal: str) -> list[Step]:
    """If the user asked to save to a file but the plan has no action step, add one.

    The planner sometimes folds "and save it to x.md" into the writing step, but
    the writing agent can't touch files; only the (approval-gated) action agent can.
    """
    path = requested_file(goal)
    if not path or any(s["agent"] == "action" for s in steps):
        return steps
    steps = steps[: MAX_STEPS - 1]
    instruction = f"Save the answer from the previous step to {path}"
    return [*steps, Step(id=len(steps) + 1, agent="action", instruction=instruction)]


def ends_well(steps: list[PlanStep]) -> bool:
    """A plan ends with a writing step, or with writing then one action (e.g. save the report)."""
    if steps[-1].agent == "writing":
        return True
    return len(steps) >= 2 and steps[-1].agent == "action" and steps[-2].agent == "writing"


def normalize_plan(plan: Plan) -> list[Step]:
    """Enforce the rules the model may have ignored: step limit and how a plan ends."""
    steps = plan.steps[:MAX_STEPS]
    if not ends_well(steps):
        steps = steps[: MAX_STEPS - 1]
        steps.append(PlanStep(agent="writing", instruction=FINAL_WRITING_INSTRUCTION))
    return [Step(id=i, agent=s.agent, instruction=s.instruction) for i, s in enumerate(steps, 1)]
