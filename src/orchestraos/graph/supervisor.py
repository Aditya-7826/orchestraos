"""The supervisor graph.

    START -> recall_memory -> triage -> plan -> (next step's agent)* -> finalize
          -> store_memory -> END

After `plan` and after every agent node, `route_next_step` reads
`plan[current_step].agent` and jumps to that agent's node, or to `finalize`
when the plan is done. With a checkpointer, state is saved after every node, so
a run interrupted mid-plan resumes at the step it stopped on.
"""

import logging
from collections.abc import Mapping

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orchestraos.agents.base import Agent, AgentState
from orchestraos.graph import planner
from orchestraos.graph.state import AGENT_NAMES, StepResult, SupervisorState
from orchestraos.llm.router import ModelRouter
from orchestraos.memory.base import Memory

logger = logging.getLogger(__name__)

# Earlier results are passed to later agents; cap each so small models' context isn't blown.
MAX_RESULT_CHARS = 3000


def format_results(results: list[StepResult]) -> str:
    """Earlier steps' results as context for the next agent."""
    blocks = []
    for r in results:
        status = "" if r["ok"] else " (FAILED)"
        output = r["output"][:MAX_RESULT_CHARS]
        block = f"### Step {r['step_id']} — {r['agent']}{status}\n{output}"
        if r["sources"]:
            block += "\nSources: " + ", ".join(r["sources"])
        if r["artifacts"]:
            block += "\nFiles: " + ", ".join(r["artifacts"])
        blocks.append(block)
    return "\n\n".join(blocks)


def route_next_step(state: SupervisorState) -> str:
    plan, index = state["plan"], state.get("current_step", 0)
    return plan[index]["agent"] if index < len(plan) else "finalize"


def build_supervisor(
    router: ModelRouter,
    agents: Mapping[str, Agent],
    memory: Memory,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    missing = set(AGENT_NAMES) - set(agents)
    if missing:
        raise ValueError(f"no agent registered for: {sorted(missing)}")

    async def recall_memory(state: SupervisorState) -> SupervisorState:
        return {
            "memory_context": await memory.recall(state["goal"]),
            "messages": [HumanMessage(state["goal"])],
            "current_step": 0,
        }

    async def triage(state: SupervisorState) -> SupervisorState:
        return {
            "mode": await planner.triage(router, state["goal"], state.get("memory_context", ""))
        }

    async def plan(state: SupervisorState) -> SupervisorState:
        goal, memory_context = state["goal"], state.get("memory_context", "")
        if state["mode"] == "direct":
            steps = planner.direct_plan(goal)
        else:
            steps = await planner.make_plan(router, goal, memory_context)
        logger.info("plan: %s", [(s["agent"], s["instruction"]) for s in steps])
        return {"plan": steps}

    def make_agent_node(name: str, agent: Agent):
        async def run_agent(state: SupervisorState) -> SupervisorState:
            step = state["plan"][state["current_step"]]
            task: AgentState = {
                "goal": state["goal"],
                "instruction": step["instruction"],
                "context": format_results(state.get("results", [])),
                "memory": state.get("memory_context", ""),
            }
            try:
                out = await agent.ainvoke(task)
                result = StepResult(
                    step_id=step["id"],
                    agent=name,
                    ok=True,
                    output=out.get("output", ""),
                    sources=out.get("sources", []),
                    artifacts=out.get("artifacts", []),
                )
            except Exception as exc:  # one failing agent shouldn't sink the whole run
                logger.exception("%s agent failed on step %s", name, step["id"])
                result = StepResult(
                    step_id=step["id"],
                    agent=name,
                    ok=False,
                    output=f"Error: {exc}",
                    sources=[],
                    artifacts=[],
                )
            return {"results": [result], "current_step": state["current_step"] + 1}

        run_agent.__name__ = f"run_{name}"
        return run_agent

    def finalize(state: SupervisorState) -> SupervisorState:
        results = state.get("results", [])
        writing = [r for r in results if r["agent"] == "writing" and r["ok"]]
        answer = writing[-1]["output"] if writing else format_results(results)
        # Every source any step used, de-duplicated, in first-seen order.
        sources = list(dict.fromkeys(s for r in results for s in r["sources"]))
        return {"final_answer": answer, "sources": sources, "messages": [AIMessage(answer)]}

    async def store_memory(state: SupervisorState) -> SupervisorState:
        await memory.remember(state["goal"], state["final_answer"])
        return {}

    graph = StateGraph(SupervisorState)
    graph.add_node("recall_memory", recall_memory)
    graph.add_node("triage", triage)
    graph.add_node("plan", plan)
    for name in AGENT_NAMES:
        graph.add_node(name, make_agent_node(name, agents[name]))
    graph.add_node("finalize", finalize)
    graph.add_node("store_memory", store_memory)

    targets = [*AGENT_NAMES, "finalize"]
    graph.add_edge(START, "recall_memory")
    graph.add_edge("recall_memory", "triage")
    graph.add_edge("triage", "plan")
    graph.add_conditional_edges("plan", route_next_step, targets)
    for name in AGENT_NAMES:
        graph.add_conditional_edges(name, route_next_step, targets)
    graph.add_edge("finalize", "store_memory")
    graph.add_edge("store_memory", END)

    return graph.compile(checkpointer=checkpointer, name="supervisor")
