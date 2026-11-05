"""Typed state shared by the supervisor graph.

Fields with an `Annotated[..., reducer]` type are *appended to* when a node
returns them; every other field is *replaced*. That is how several agent nodes
can each add a result without overwriting each other.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

AgentName = Literal["research", "writing", "analytics", "action"]
AGENT_NAMES: tuple[AgentName, ...] = ("research", "writing", "analytics", "action")


class Step(TypedDict):
    id: int
    agent: AgentName
    instruction: str


class StepResult(TypedDict):
    step_id: int
    agent: str
    ok: bool
    output: str
    sources: list[str]  # URLs / file paths the output is based on
    artifacts: list[str]  # files produced (e.g. charts), relative to the workspace


class Approval(TypedDict):
    step_id: int
    action: str
    approved: bool
    note: str


class SupervisorState(TypedDict, total=False):
    goal: str
    memory_context: str  # what the memory agent recalled before planning
    mode: Literal["direct", "plan"]  # triage decision
    plan: list[Step]
    current_step: int  # index into `plan` of the next step to run
    results: Annotated[list[StepResult], operator.add]
    approvals: Annotated[list[Approval], operator.add]
    messages: Annotated[list[AnyMessage], add_messages]
    final_answer: str
    sources: list[str]
