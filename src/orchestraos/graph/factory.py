"""Wires the real components together. Tests build the supervisor with fakes instead."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.graph.state import CompiledStateGraph

from orchestraos.agents.action import build_action_agent
from orchestraos.agents.analytics import build_analytics_agent
from orchestraos.agents.base import Agent
from orchestraos.agents.research import build_research_agent
from orchestraos.agents.research_a2a import build_a2a_research_agent
from orchestraos.agents.writing import build_writing_agent
from orchestraos.config import Settings
from orchestraos.graph.checkpoint import open_checkpointer
from orchestraos.graph.supervisor import build_supervisor
from orchestraos.llm.router import ModelRouter
from orchestraos.memory.agent import MemoryAgent
from orchestraos.memory.base import Memory, NullMemory
from orchestraos.memory.graph import GraphStore, Neo4jGraph, NullGraph
from orchestraos.memory.vector import OllamaEmbedder, VectorMemory
from orchestraos.observability import flush, traced
from orchestraos.tools.client import ToolCaller, open_toolset, restrict

logger = logging.getLogger(__name__)


def build_agents(
    router: ModelRouter, tools: ToolCaller, settings: Settings | None = None
) -> dict[str, Agent]:
    """Each agent only gets the tools it needs. Only the action agent can write files,
    and it only does so after the user approves (see agents/action.py)."""
    read_only = ("list_files", "read_file")
    if settings and settings.research_mode == "a2a":
        # Same agent, different transport: it runs as its own service (a2a/research_service.py).
        research = build_a2a_research_agent(settings.a2a_research_url)
    else:
        research = build_research_agent(
            router, restrict(tools, "web_search", "fetch_page", *read_only)
        )
    return {
        "research": research,
        "analytics": build_analytics_agent(router, restrict(tools, "run_python", *read_only)),
        "writing": build_writing_agent(router),
        "action": build_action_agent(router, restrict(tools, "write_file", "list_files")),
    }


@asynccontextmanager
async def open_memory(settings: Settings, router: ModelRouter) -> AsyncIterator[Memory]:
    """Semantic memory always (when enabled); graph memory only if Neo4j is configured and up."""
    if not settings.memory_enabled:
        yield NullMemory()
        return

    graph: GraphStore = NullGraph()
    if settings.neo4j_uri and settings.neo4j_password:
        try:
            graph = await Neo4jGraph.connect(
                settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
            )
        except Exception as exc:  # the app is still useful without the graph
            logger.warning("Neo4j unavailable (%s); using semantic memory only", exc)

    embedder = OllamaEmbedder(settings.embed_model, settings.ollama_base_url)
    vector = VectorMemory(settings.data_dir / "chroma", embedder)
    try:
        yield MemoryAgent(router, vector, graph)
    finally:
        await graph.close()


@asynccontextmanager
async def open_app(settings: Settings) -> AsyncIterator[CompiledStateGraph]:
    """The full supervisor app. MCP servers, DB connections etc. are closed on exit."""
    router = ModelRouter.from_settings(settings)
    async with (
        open_toolset(settings) as toolset,
        open_checkpointer(settings.checkpoint_db) as checkpointer,
        open_memory(settings, router) as memory,
    ):
        graph = build_supervisor(
            router=router,
            agents=build_agents(router, toolset, settings),
            memory=memory,
            checkpointer=checkpointer,
        )
        try:
            yield traced(graph, settings)
        finally:
            flush(settings)
