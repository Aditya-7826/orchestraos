"""The agent bar's states and status line, worked out from a run's events."""

from ui.agent_status import agent_board, board_html

PLAN = {
    "type": "plan",
    "steps": [
        {"id": 1, "agent": "analytics", "instruction": "Chart revenue per region"},
        {"id": 2, "agent": "writing", "instruction": "Present the results"},
        {"id": 3, "agent": "action", "instruction": "Save to reports/r.md"},
    ],
}


def states(board) -> dict[str, str]:
    return {card.agent: card.state for card in board.cards}


def step(step_id: int, agent: str, ok: bool = True) -> dict:
    return {"type": "step", "step_id": step_id, "agent": agent, "ok": ok}


def activity(agent: str, text: str) -> dict:
    return {"type": "activity", "agent": agent, "text": text}


def test_before_any_event_everything_is_idle():
    board = agent_board([])
    assert set(states(board).values()) == {"idle"}
    assert (board.label, board.status) == ("ready", "Waiting for a goal")


def test_planning_phase_has_no_running_agent():
    board = agent_board([{"type": "triage", "mode": "plan"}])
    assert "running" not in states(board).values()
    assert (board.label, board.status) == ("planner", "Planning the steps")


def test_first_step_runs_and_shows_its_latest_activity():
    board = agent_board([PLAN, activity("analytics", "Profiling sales.csv")])
    assert states(board) == {
        "research": "idle",
        "analytics": "running",
        "writing": "queued",
        "action": "queued",
    }
    assert (board.label, board.status) == ("analytics", "Profiling sales.csv")


def test_without_activity_the_running_agent_shows_its_instruction():
    board = agent_board([PLAN])
    assert board.status == "Chart revenue per region"


def test_finished_steps_are_done_and_the_next_agent_runs():
    board = agent_board([PLAN, activity("analytics", "Running code"), step(1, "analytics")])
    assert states(board)["analytics"] == "done"
    assert states(board)["writing"] == "running"
    assert board.status == "Present the results"  # analytics' old line doesn't leak over


def test_approval_pause_marks_action_as_waiting_until_resumed():
    paused = [
        PLAN,
        step(1, "analytics"),
        step(2, "writing"),
        activity("action", "Waiting for your approval to write reports/r.md"),
        {"type": "interrupt", "value": {}},
        {"type": "done", "status": "interrupted"},
    ]
    board = agent_board(paused)
    assert states(board)["action"] == "waiting"
    assert board.status == "Waiting for your approval to write reports/r.md"

    resumed = agent_board([*paused, activity("action", "Approved: writing reports/r.md")])
    assert states(resumed)["action"] == "running"
    assert resumed.status == "Approved: writing reports/r.md"


def test_completed_run_marks_used_agents_done():
    events = [
        PLAN,
        step(1, "analytics"),
        step(2, "writing"),
        step(3, "action"),
        {"type": "final", "answer": "x"},
        {"type": "done", "status": "completed"},
    ]
    board = agent_board(events)
    assert states(board) == {
        "research": "idle",
        "analytics": "done",
        "writing": "done",
        "action": "done",
    }
    assert board.label == "done"


def test_failed_step_and_error_are_shown():
    events = [PLAN, step(1, "analytics", ok=False)]
    assert states(agent_board(events))["analytics"] == "failed"

    crashed = agent_board([PLAN, {"type": "error", "message": "boom"}, {"type": "done"}])
    assert (crashed.label, crashed.status) == ("error", "boom")
    assert "running" not in states(crashed).values()


def test_html_escapes_model_text():
    html = board_html(agent_board([PLAN, activity("analytics", "<script>x</script>")]))
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert html.count('class="agent ') == 4
