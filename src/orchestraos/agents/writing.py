"""Writing agent: turns earlier steps' results into the markdown answer.

START -> draft -> END
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from orchestraos.agents.base import Agent, AgentState
from orchestraos.agents.prompts import WRITING_SYSTEM
from orchestraos.llm.router import ModelRouter, TaskKind


def writing_prompt(state: AgentState) -> str:
    parts = [f"User's goal: {state['goal']}", f"Your instruction: {state['instruction']}"]
    if memory := state.get("memory"):
        parts.append(f"What you know about the user:\n{memory}")
    parts.append(f"Results from earlier steps:\n\n{state.get('context') or '(none)'}")
    return "\n\n".join(parts)


def build_writing_agent(router: ModelRouter) -> Agent:
    async def draft(state: AgentState) -> AgentState:
        messages = [SystemMessage(WRITING_SYSTEM), HumanMessage(writing_prompt(state))]
        reply = await router.ainvoke(TaskKind.WRITE, messages)
        return {"output": reply.text.strip(), "sources": [], "artifacts": []}

    graph = StateGraph(AgentState)
    graph.add_node("draft", draft)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", END)
    return graph.compile(name="writing_agent")
