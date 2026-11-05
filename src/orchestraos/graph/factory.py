"""Wires the real components together. Tests build the supervisor with fakes instead."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.graph.state import CompiledStateGraph

from orchestraos.agents.base import build_stub_agent
from orchestraos.config import Settings
from orchestraos.graph.checkpoint import open_checkpointer
from orchestraos.graph.state import AGENT_NAMES
from orchestraos.graph.supervisor import build_supervisor
from orchestraos.llm.router import ModelRouter
from orchestraos.memory.base import NullMemory


@asynccontextmanager
async def open_app(settings: Settings) -> AsyncIterator[CompiledStateGraph]:
    """The full supervisor app. Resources (checkpoint DB, ...) are closed on exit."""
    router = ModelRouter.from_settings(settings)
    agents = {name: build_stub_agent(name) for name in AGENT_NAMES}
    async with open_checkpointer(settings.checkpoint_db) as checkpointer:
        yield build_supervisor(
            router=router, agents=agents, memory=NullMemory(), checkpointer=checkpointer
        )
