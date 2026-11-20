"""Which agent is working, and on what, worked out from a run's events.

The API streams `plan`, `activity` (an agent's one-line status), `step` (a step
finished), `interrupt`, `final` and `done` events. `agent_board()` folds them into
one card per agent plus the page's status line; `board_html()` draws them.
Pure functions, so they're tested without Streamlit.
"""

from dataclasses import dataclass
from html import escape

AGENTS = ("research", "analytics", "writing", "action")
ICONS = {"research": "🔎", "analytics": "📊", "writing": "✍️", "action": "🛠️"}
ROLES = {
    "research": "Searches the web and reads documents",
    "analytics": "Runs pandas code in a sandbox",
    "writing": "Writes the markdown answer",
    "action": "Saves files after you approve",
}
STATE_LABELS = {
    "idle": "idle",
    "queued": "queued",
    "running": "running",
    "waiting": "needs you",
    "done": "done",
    "failed": "failed",
}


@dataclass
class Card:
    agent: str
    state: str  # one of STATE_LABELS
    line: str


@dataclass
class Board:
    cards: list[Card]
    label: str  # who the status line is about, e.g. "analytics" or "planner"
    status: str  # the status line itself


def agent_board(events: list[dict], finished: bool = False) -> Board:
    plan: list[dict] = []
    results: dict[int, bool] = {}  # step id -> ok
    activity: dict[str, str] = {}  # agent -> latest line, for the step it's on now
    triaged = waiting = final = False
    error = ""
    status = ""  # the run's own status from the "done" event

    for event in events:
        kind = event.get("type")
        if status == "interrupted" and kind != "done":  # resumed after the user answered
            status, waiting = "", False
        if kind == "triage":
            triaged = True
        elif kind == "plan":
            plan = event["steps"]
        elif kind == "activity":
            activity[event["agent"]] = event["text"]
        elif kind == "step":
            results[event["step_id"]] = event["ok"]
            activity.pop(event["agent"], None)  # its next step starts with a fresh line
            waiting = False
        elif kind == "interrupt":
            waiting = True
        elif kind == "final":
            final = True
        elif kind == "error":
            error = event.get("message", "error")
        elif kind == "done":
            status = event.get("status", "")
            if status == "completed":
                finished = True

    current = next((s for s in plan if s["id"] not in results), None)
    if finished or error or (status and status != "interrupted"):
        current = None

    cards = []
    for agent in AGENTS:
        steps = [s for s in plan if s["agent"] == agent]
        if current and current["agent"] == agent:
            state = "waiting" if waiting else "running"
            line = activity.get(agent) or current["instruction"]
        elif any(s["id"] not in results for s in steps) and not finished and not error:
            state = "queued"
            next_step = next(s for s in steps if s["id"] not in results)
            line = f"Step {next_step['id']}: {next_step['instruction']}"
        elif any(results.get(s["id"]) is False for s in steps):
            state, line = "failed", "A step failed; see How I did it"
        elif steps:
            done_ids = ", ".join(str(s["id"]) for s in steps)
            state, line = "done", f"Finished step {done_ids}"
        else:
            state, line = "idle", ROLES[agent]
        cards.append(Card(agent, state, line))

    if error:
        label, line = "error", error
    elif current:
        label = current["agent"]
        line = cards[AGENTS.index(label)].line
    elif finished or final:
        label, line = "done", "Finished. The answer is below."
    elif plan:
        label, line = "planner", "Putting the answer together"
    elif triaged:
        label, line = "planner", "Planning the steps"
    elif events:
        label, line = "planner", "Recalling memory and deciding how to answer"
    else:
        label, line = "ready", "Waiting for a goal"
    return Board(cards, label, line)


def board_html(board: Board) -> str:
    cards = "".join(
        f'<div class="agent {c.state}" title="{escape(ROLES[c.agent])}">'
        f'<div class="agent-head"><span class="agent-dot"></span>'
        f'<span class="agent-name">{ICONS[c.agent]} {c.agent}</span>'
        f'<span class="agent-state">{STATE_LABELS[c.state]}</span></div>'
        f'<div class="agent-line" title="{escape(c.line)}">{escape(c.line)}</div></div>'
        for c in board.cards
    )
    status = (
        f'<div class="status-line"><b>{escape(board.label)}</b>'
        f'<span title="{escape(board.status)}">{escape(board.status)}</span></div>'
    )
    return f'<div class="agent-bar">{cards}</div>{status}'
