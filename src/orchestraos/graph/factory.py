"""Wires the real components together. Tests build the supervisor with fakes instead."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.graph.state import CompiledStateGraph

from orchestraos.agents.analytics import build_analytics_agent
from orchestraos.agents.base import Agent, build_stub_agent
from orchestraos.agents.research import build_research_agent
from orchestraos.agents.writing import build_writing_agent
from orchestraos.config import Settings
from orchestraos.graph.checkpoint import open_checkpointer
from orchestraos.graph.supervisor import build_supervisor
from orchestraos.llm.router import ModelRouter
from orchestraos.memory.base import NullMemory
from orchestraos.tools.client import ToolCaller, open_toolset


def build_agents(router: ModelRouter, tools: ToolCaller) -> dict[str, Agent]:
    return {
        "research": build_research_agent(router, tools),
        "analytics": build_analytics_agent(router, tools),
        "writing": build_writing_agent(router),
        "action": build_stub_agent("action"),
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
