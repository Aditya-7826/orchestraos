"""The research agent served over A2A, so any A2A client (not just OrchestraOS) can use it.

    uv run uvicorn orchestraos.a2a.research_service:app --port 8100
    curl http://localhost:8100/.well-known/agent-card.json

It wraps the *same* research subgraph the supervisor uses in-process; only the
transport differs. Each request becomes an A2A Task: submitted -> working ->
artifact ("findings") -> completed (or failed, with the error as the message).
"""

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from a2a.helpers import new_task_from_user_message, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils import DEFAULT_RPC_URL
from fastapi import FastAPI

from orchestraos import __version__
from orchestraos.a2a.protocol import ARTIFACT_NAME, read_task, result_parts
from orchestraos.agents.base import Agent

logger = logging.getLogger(__name__)

AgentFactory = Callable[[], AbstractAsyncContextManager[Agent]]


def agent_card(public_url: str) -> AgentCard:
    return AgentCard(
        name="OrchestraOS Research Agent",
        description=(
            "Researches a question on the web (DuckDuckGo) and in local documents, and "
            "returns concise findings in markdown with numbered citations and source URLs. "
            "Runs entirely on local models."
        ),
        version=__version__,
        supported_interfaces=[
            AgentInterface(url=public_url, protocol_binding="JSONRPC", protocol_version="1.0")
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["application/json", "text/plain"],
        default_output_modes=["text/markdown", "application/json"],
        skills=[
            AgentSkill(
                id="web_research",
                name="Web research with sources",
                description="Search the web, read the top pages, and summarise with citations.",
                tags=["research", "search", "citations"],
                examples=["Compare LangGraph and CrewAI for building multi-agent systems"],
            )
        ],
    )


class ResearchExecutor(AgentExecutor):
    def __init__(self, agent: Agent | None = None):
        self.agent = agent  # set at startup, once the MCP tool servers are running

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if self.agent is None:
            raise RuntimeError("research agent not started")
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        try:
            out = await self.agent.ainvoke({**read_task(context.message), "memory": ""})
        except Exception as exc:
            logger.exception("research task %s failed", task.id)
            message = new_text_message(f"{type(exc).__name__}: {exc}", context_id=task.context_id,
                                       task_id=task.id)  # fmt: skip
            await updater.failed(message)
            return
        await updater.add_artifact(
            result_parts(out.get("output", ""), out.get("sources", [])), name=ARTIFACT_NAME
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("research tasks can't be cancelled")


def create_app(open_agent: AgentFactory | None = None, public_url: str | None = None) -> FastAPI:
    """`open_agent` provides the research subgraph (default: the real one); tests pass fakes."""
    url = public_url or os.environ.get("A2A_PUBLIC_URL", "http://localhost:8100/")
    card = agent_card(url)
    executor = ResearchExecutor()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with (open_agent or open_real_agent)() as agent:
            executor.agent = agent
            yield
            executor.agent = None

    app = FastAPI(title="OrchestraOS Research Agent (A2A)", version=__version__, lifespan=lifespan)
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),  # GET /.well-known/agent-card.json
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=DEFAULT_RPC_URL),  # POST /
    )
    return app


@asynccontextmanager
async def open_real_agent() -> AsyncIterator[Agent]:
    from orchestraos.agents.research import build_research_agent
    from orchestraos.config import get_settings
    from orchestraos.llm.router import ModelRouter
    from orchestraos.observability import flush, traced
    from orchestraos.tools.client import open_toolset, restrict

    settings = get_settings()
    router = ModelRouter.from_settings(settings)
    async with open_toolset(settings, servers=["search", "filesystem"]) as toolset:
        tools = restrict(toolset, "web_search", "fetch_page", "list_files", "read_file")
        try:
            yield traced(build_research_agent(router, tools), settings)
        finally:
            flush(settings)


app = create_app()
