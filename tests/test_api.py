import json
from contextlib import asynccontextmanager

import pytest
from fakes import fake_router, system_prompt
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from orchestraos.agents.base import AgentState, build_function_agent, build_stub_agent
from orchestraos.api.events import to_events
from orchestraos.api.main import create_app
from orchestraos.graph.state import AGENT_NAMES
from orchestraos.graph.supervisor import build_supervisor
from orchestraos.memory.base import NullMemory

PLAN = [{"agent": "analytics", "instruction": "crunch"}, {"agent": "writing", "instruction": "w"}]


def respond(messages) -> str:
    if "decide how to handle" in system_prompt(messages):
        return json.dumps({"mode": "plan"})
    return json.dumps({"steps": PLAN})


def make_client(tmp_path, analytics=None, memory=None) -> TestClient:
    def default_analytics(task: AgentState) -> AgentState:
        return {"output": "total 40", "sources": ["sales.csv"], "artifacts": ["outputs/a/c.png"]}

    agents = {name: build_stub_agent(name) for name in AGENT_NAMES}
    agents["analytics"] = build_function_agent("analytics", analytics or default_analytics)

    @asynccontextmanager
    async def open_graph():
        yield build_supervisor(
            fake_router(respond), agents, memory or NullMemory(), InMemorySaver()
        )

    return TestClient(create_app(open_graph=open_graph, workspace_dir=tmp_path))


def read_events(client: TestClient, thread_id: str, after: int = 0) -> list[dict]:
    events = []
    with client.stream("GET", f"/runs/{thread_id}/events", params={"after": after}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line.removeprefix("data:").strip()))
    return events


def test_run_streams_plan_steps_and_final_answer(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/runs", json={"goal": "analyse sales"})
        assert response.status_code == 202
        thread_id = response.json()["thread_id"]

        events = read_events(client, thread_id)
        types = [e["type"] for e in events]
        assert types == ["triage", "plan", "step", "step", "final", "done"]
        assert events[2]["agent"] == "analytics" and events[2]["artifacts"] == ["outputs/a/c.png"]
        final = events[4]
        assert final["sources"] == ["sales.csv"]
        assert final["answer"].endswith("![chart](outputs/a/c.png)")  # chart appended by code
        assert events[-1] == {"type": "done", "status": "completed"}


def test_events_can_be_replayed_from_an_index(tmp_path):
    with make_client(tmp_path) as client:
        thread_id = client.post("/runs", json={"goal": "g"}).json()["thread_id"]
        everything = read_events(client, thread_id)
        assert read_events(client, thread_id, after=4) == everything[4:]


def test_run_state_is_available_after_completion(tmp_path):
    with make_client(tmp_path) as client:
        thread_id = client.post("/runs", json={"goal": "g"}).json()["thread_id"]
        read_events(client, thread_id)
        state = client.get(f"/runs/{thread_id}").json()
        assert state["status"] == "completed"
        assert [s["agent"] for s in state["plan"]] == ["analytics", "writing"]
        assert state["artifacts"] == ["outputs/a/c.png"]


def test_unknown_run_is_404(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/runs/nope").status_code == 404
        assert client.get("/runs/nope/events").status_code == 404


def test_empty_goal_is_rejected(tmp_path):
    with make_client(tmp_path) as client:
        assert client.post("/runs", json={"goal": ""}).status_code == 422


def test_finished_run_cannot_be_resumed(tmp_path):
    with make_client(tmp_path) as client:
        thread_id = client.post("/runs", json={"goal": "g"}).json()["thread_id"]
        read_events(client, thread_id)
        assert client.post(f"/runs/{thread_id}/resume").status_code == 409


class FlakyMemory(NullMemory):
    """Fails the first time it stores, like a database blip at the very end of a run."""

    def __init__(self):
        self.failed = False

    async def remember(self, goal: str, answer: str) -> None:
        if not self.failed:
            self.failed = True
            raise ConnectionError("memory store unavailable")


def test_failed_run_reports_error_and_can_be_resumed(tmp_path):
    with make_client(tmp_path, memory=FlakyMemory()) as client:
        thread_id = client.post("/runs", json={"goal": "g"}).json()["thread_id"]
        events = read_events(client, thread_id)
        assert events[-2]["type"] == "error" and "memory store unavailable" in events[-2]["message"]
        assert events[-1] == {"type": "done", "status": "failed"}
        assert client.get(f"/runs/{thread_id}").json()["status"] == "failed"

        assert client.post(f"/runs/{thread_id}/resume").status_code == 202
        resumed = read_events(client, thread_id, after=len(events))
        assert resumed[-1] == {"type": "done", "status": "completed"}
        assert client.get(f"/runs/{thread_id}").json()["status"] == "completed"


def test_failing_agent_step_is_reported_not_fatal(tmp_path):
    def broken(task: AgentState) -> AgentState:
        raise RuntimeError("container died")

    with make_client(tmp_path, analytics=broken) as client:
        thread_id = client.post("/runs", json={"goal": "g"}).json()["thread_id"]
        events = read_events(client, thread_id)
        step = next(e for e in events if e["type"] == "step")
        assert step["ok"] is False and "container died" in step["preview"]
        assert events[-1] == {"type": "done", "status": "completed"}


# --- files ------------------------------------------------------------------


def test_upload_list_and_download_files(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/files", files={"upload": ("sales.csv", b"a,b\n1,2\n")})
        assert response.status_code == 201 and response.json() == {"path": "sales.csv"}
        assert client.get("/files").json() == ["sales.csv"]
        assert client.get("/files/sales.csv").content == b"a,b\n1,2\n"


@pytest.mark.parametrize("name", ["../evil.csv", "..\\evil.csv", "sub/../../evil.csv"])
def test_upload_cannot_escape_workspace(tmp_path, name):
    with make_client(tmp_path / "ws") as client:
        response = client.post("/files", files={"upload": (name, b"x")})
        assert response.status_code in (201, 400)
        assert not (tmp_path / "evil.csv").exists()


def test_download_outside_workspace_is_rejected(tmp_path):
    with make_client(tmp_path / "ws") as client:
        assert client.get("/files/..%2Fsecret.txt").status_code in (400, 404)
        assert client.get("/files/missing.png").status_code == 404


# --- event conversion -------------------------------------------------------


def test_only_writing_agent_tokens_are_streamed():
    token = (AIMessageChunk("Hel"), {})
    assert to_events(("writing:abc",), "messages", token) == [{"type": "token", "text": "Hel"}]
    assert to_events(("research:abc",), "messages", token) == []
    assert to_events((), "messages", token) == []


def test_inner_subgraph_updates_are_dropped():
    assert to_events(("analytics:abc",), "updates", {"run_code": {"run": {}}}) == []
