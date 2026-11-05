"""Triage (small model) and planning (large model).

Both use `ModelRouter.invoke_json`, so a bad answer is retried and falls back
to the other model. If even that fails we don't crash the run: triage defaults
to "plan" (the safer choice) and planning defaults to a single writing step.
"""

import logging
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


def triage(router: ModelRouter, goal: str, memory_context: str = "") -> Literal["direct", "plan"]:
    messages = [SystemMessage(TRIAGE_SYSTEM), HumanMessage(with_memory(goal, memory_context))]
    try:
        result = router.invoke_json(TaskKind.CLASSIFY, messages, Triage)
    except ModelRouterError:
        logger.warning("triage failed; defaulting to 'plan'")
        return "plan"
    return result.parsed.mode


def direct_plan(goal: str) -> list[Step]:
    return [Step(id=1, agent="writing", instruction=goal)]


def make_plan(router: ModelRouter, goal: str, memory_context: str = "") -> list[Step]:
    messages = [
        SystemMessage(PLANNER_SYSTEM.format(max_steps=MAX_STEPS)),
        HumanMessage(with_memory(goal, memory_context)),
    ]
    try:
        result = router.invoke_json(TaskKind.PLAN, messages, Plan)
    except ModelRouterError:
        logger.warning("planning failed; falling back to a single writing step")
        return direct_plan(goal)
    return normalize_plan(result.parsed)


def normalize_plan(plan: Plan) -> list[Step]:
    """Enforce the rules the model may have ignored: step limit and a final writing step."""
    steps = plan.steps[:MAX_STEPS]
    if steps[-1].agent != "writing":
        steps = steps[: MAX_STEPS - 1]
        steps.append(PlanStep(agent="writing", instruction=FINAL_WRITING_INSTRUCTION))
    return [Step(id=i, agent=s.agent, instruction=s.instruction) for i, s in enumerate(steps, 1)]
