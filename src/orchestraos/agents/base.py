"""What every specialist agent looks like to the supervisor.

Each agent is its own compiled LangGraph subgraph over `AgentState`: the
supervisor fills in the task fields, the agent fills in the output fields.
Keeping the contract this small means agents can be built, tested and swapped
(e.g. local research vs research over A2A) independently of the supervisor.
"""

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orchestraos.graph.state import StepResult


class AgentState(TypedDict, total=False):
    # --- task (set by the supervisor) ---
    goal: str  # the user's overall goal
    instruction: str  # this step's instruction from the plan
    context: str  # formatted results of earlier steps
    memory: str  # recalled memory context
    results: list[StepResult]  # earlier steps' raw results (e.g. to save one to a file)
    # --- output (set by the agent) ---
    output: str
    sources: list[str]
    artifacts: list[str]
    approval: dict  # set by agents that asked the user: {action, approved, note}


Agent = CompiledStateGraph


def build_function_agent(name: str, fn: Callable[[AgentState], AgentState]) -> Agent:
    """Wrap a plain function as a one-node agent subgraph."""
    graph = StateGraph(AgentState)
    graph.add_node("run", fn)
    graph.add_edge(START, "run")
    graph.add_edge("run", END)
    return graph.compile(name=f"{name}_agent")


def build_stub_agent(name: str) -> Agent:
    """A placeholder agent that echoes its instruction. Replaced by real agents later."""

    def run(state: AgentState) -> AgentState:
        return {"output": f"[{name} stub] {state['instruction']}", "sources": [], "artifacts": []}

    return build_function_agent(name, run)
