"""Wires the real components together. Tests build the supervisor with fakes instead."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.graph.state import CompiledStateGraph

from orchestraos.agents.action import build_action_agent
from orchestraos.agents.analytics import build_analytics_agent
from orchestraos.agents.base import Agent
from orchestraos.agents.research import build_research_agent
from orchestraos.agents.writing import build_writing_agent
from orchestraos.config import Settings
from orchestraos.graph.checkpoint import open_checkpointer
from orchestraos.graph.supervisor import build_supervisor
from orchestraos.llm.router import ModelRouter
from orchestraos.memory.base import NullMemory
from orchestraos.tools.client import ToolCaller, open_toolset, restrict


def build_agents(router: ModelRouter, tools: ToolCaller) -> dict[str, Agent]:
    """Each agent only gets the tools it needs. Only the action agent can write files,
    and it only does so after the user approves (see agents/action.py)."""
    read_only = ("list_files", "read_file")
    return {
        "research": build_research_agent(
            router, restrict(tools, "web_search", "fetch_page", *read_only)
        ),
        "analytics": build_analytics_agent(router, restrict(tools, "run_python", *read_only)),
        "writing": build_writing_agent(router),
        "action": build_action_agent(router, restrict(tools, "write_file", "list_files")),
    }


@asynccontextmanager
async def open_app(settings: Settings) -> AsyncIterator[CompiledStateGraph]:
    """The full supervisor app. MCP servers and the checkpoint DB are closed on exit."""
    router = ModelRouter.from_settings(settings)
    async with (
        open_toolset(settings) as toolset,
        open_checkpointer(settings.checkpoint_db) as checkpointer,
    ):
        yield build_supervisor(
            router=router,
            agents=build_agents(router, toolset),
            memory=NullMemory(),
            checkpointer=checkpointer,
        )
