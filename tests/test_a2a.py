"""A2A: the research service and the client the supervisor uses, talking in-process."""

from contextlib import asynccontextmanager

import httpx
import pytest
from a2a.helpers import new_text_part
from a2a.types import Message, Role
from fastapi.testclient import TestClient

from orchestraos.a2a.protocol import read_task, task_message
from orchestraos.a2a.research_service import create_app
from orchestraos.agents.base import AgentState, build_function_agent
from orchestraos.agents.common import AgentError
from orchestraos.agents.research_a2a import build_a2a_research_agent

BASE = "http://testserver/"


def service(research):
    @asynccontextmanager
    async def open_agent():
        yield build_function_agent("research", research)

    return create_app(open_agent=open_agent, public_url=BASE)


@asynccontextmanager
async def running(app):
    """Start the app (lifespan) and return an A2A research agent wired straight into it."""
    async with app.router.lifespan_context(app):

        def http_client():
            return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)

        yield build_a2a_research_agent(BASE, http_client=http_client)


TASK = {"goal": "pick a framework", "instruction": "Compare LangGraph and CrewAI",
        "context": "", "memory": ""}  # fmt: skip


def test_agent_card_is_served_at_the_well_known_path():
    with TestClient(service(lambda task: {"output": ""})) as client:
        card = client.get("/.well-known/agent-card.json").json()
    assert card["name"] == "OrchestraOS Research Agent"
    assert card["skills"][0]["id"] == "web_research"
    assert card["supportedInterfaces"][0]["url"] == BASE
    assert card["capabilities"]["streaming"] is True


async def test_supervisor_side_agent_gets_findings_and_sources_over_a2a():
    seen = {}

    def research(task: AgentState) -> AgentState:
        seen.update(task)
        return {"output": "LangGraph is lower level [1].", "sources": ["https://a.example"]}

    async with running(service(research)) as remote:
        out = await remote.ainvoke(TASK)

    assert out["output"] == "LangGraph is lower level [1]."
    assert out["sources"] == ["https://a.example"]
    assert seen["instruction"] == "Compare LangGraph and CrewAI"  # arrived intact over A2A
    assert seen["goal"] == "pick a framework"


async def test_remote_failure_becomes_an_agent_error():
    def research(task: AgentState) -> AgentState:
        raise RuntimeError("search rate-limited")

    async with running(service(research)) as remote:
        with pytest.raises(AgentError, match="TASK_STATE_FAILED.*search rate-limited"):
            await remote.ainvoke(TASK)


def test_task_message_round_trips():
    message = task_message("find X", goal="g", context="c")
    assert read_task(message) == {"instruction": "find X", "goal": "g", "context": "c"}


def test_plain_text_messages_from_other_clients_work():
    message = Message(role=Role.ROLE_USER, message_id="1", parts=[new_text_part("find Y")])
    assert read_task(message)["instruction"] == "find Y"


def test_empty_message_is_rejected():
    with pytest.raises(ValueError):
        read_task(Message(role=Role.ROLE_USER, message_id="1"))
