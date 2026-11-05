"""Approval round-trip through the HTTP API: run -> interrupt -> POST approval -> finish."""

from contextlib import asynccontextmanager

from fastapi.testclient import TestClient
from test_action import SAVE_STEP_1, llm, supervisor, workspace_tools
from test_api import read_events

from orchestraos.api.main import create_app


def make_client(tmp_path):
    tools, written = workspace_tools()

    @asynccontextmanager
    async def open_graph():
        yield supervisor(llm(SAVE_STEP_1), tools)

    return TestClient(create_app(open_graph=open_graph, workspace_dir=tmp_path)), written


def start_and_wait_for_approval(client) -> tuple[str, list[dict]]:
    thread_id = client.post("/runs", json={"goal": "report and save it"}).json()["thread_id"]
    events = read_events(client, thread_id)
    assert events[-2]["type"] == "interrupt"
    assert events[-1] == {"type": "done", "status": "interrupted"}
    return thread_id, events


def test_approve_resumes_and_writes(tmp_path):
    client, written = make_client(tmp_path)
    with client:
        thread_id, events = start_and_wait_for_approval(client)
        request = events[-2]["value"]
        assert request["actions"][0]["path"] == "reports/r.md"
        state = client.get(f"/runs/{thread_id}").json()
        assert state["status"] == "interrupted" and state["pending"] == [request]

        response = client.post(f"/runs/{thread_id}/approval", json={"approved": True})
        assert response.status_code == 202
        rest = read_events(client, thread_id, after=len(events))
        assert [e["type"] for e in rest] == ["step", "final", "done"]
        assert rest[-1]["status"] == "completed"
        assert "Saved to reports/r.md" in rest[1]["answer"]
        assert written == {"reports/r.md": "[writing stub] write the report"}


def test_reject_finishes_without_writing(tmp_path):
    client, written = make_client(tmp_path)
    with client:
        thread_id, events = start_and_wait_for_approval(client)
        client.post(f"/runs/{thread_id}/approval", json={"approved": False, "note": "no"})
        rest = read_events(client, thread_id, after=len(events))
        assert "Not saved" in rest[1]["answer"] and written == {}


def test_approval_without_pending_request_is_409(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        thread_id, events = start_and_wait_for_approval(client)
        client.post(f"/runs/{thread_id}/approval", json={"approved": True})
        read_events(client, thread_id, after=len(events))
        again = client.post(f"/runs/{thread_id}/approval", json={"approved": True})
        assert again.status_code == 409
        assert client.post("/runs/nope/approval", json={"approved": True}).status_code == 404
