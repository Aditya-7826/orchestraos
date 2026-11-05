"""OrchestraOS chat UI (Streamlit). Talks to the FastAPI backend only.

    uv run streamlit run ui/app.py

Streamlit re-runs this whole script on every interaction. A run takes minutes,
so the backend runs it in the background and this script just follows its
event stream; if the script re-runs mid-run, it reconnects and replays from the
last event it saw (`events_seen`).
"""

import json
import os
import re

import httpx
import streamlit as st
from httpx_sse import connect_sse

API = os.environ.get("ORCHESTRA_API", "http://localhost:8000")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
AGENT_ICONS = {"research": "🔎", "analytics": "📊", "writing": "✍️", "action": "🛠️"}

st.set_page_config(page_title="OrchestraOS", page_icon="🎼", layout="centered")
state = st.session_state
state.setdefault("history", [])  # finished turns
state.setdefault("active", None)  # {"thread_id", "goal", "events_seen", "events"} while running


# --- API helpers ------------------------------------------------------------


def api(method: str, path: str, **kwargs) -> httpx.Response:
    response = httpx.request(method, f"{API}{path}", timeout=30, **kwargs)
    response.raise_for_status()
    return response


@st.cache_data(show_spinner=False)
def fetch_file(path: str) -> bytes:
    return api("GET", f"/files/{path}").content


# --- rendering --------------------------------------------------------------


def render_markdown_with_images(text: str) -> None:
    """st.markdown can't load workspace-relative images, so render those with st.image."""
    position = 0
    for match in IMAGE.finditer(text):
        st.markdown(text[position : match.start()])
        try:
            st.image(fetch_file(match.group(2)), caption=match.group(1) or None)
        except httpx.HTTPError:
            st.caption(f"(missing image: {match.group(2)})")
        position = match.end()
    st.markdown(text[position:])


def step_line(event: dict) -> str:
    icon = AGENT_ICONS.get(event["agent"], "•")
    mark = "✅" if event["ok"] else "❌"
    return f"{mark} {icon} **Step {event['step_id']} · {event['agent']}**"


def render_progress(events: list[dict]) -> None:
    """The plan and each finished step, from the run's events."""
    for event in events:
        if event["type"] == "triage":
            st.caption("Answering directly" if event["mode"] == "direct" else "Making a plan…")
        elif event["type"] == "plan":
            steps = "\n".join(
                f"{s['id']}. {AGENT_ICONS.get(s['agent'], '')} **{s['agent']}**: {s['instruction']}"
                for s in event["steps"]
            )
            st.markdown(f"**Plan**\n\n{steps}")
        elif event["type"] == "step":
            st.markdown(step_line(event))
            if event["preview"]:
                st.caption(event["preview"][:300] + ("…" if len(event["preview"]) > 300 else ""))
        elif event["type"] == "error":
            st.error(event["message"])


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.markdown(turn["goal"])
    with st.chat_message("assistant"):
        with st.expander("How I did it", expanded=False):
            render_progress(turn["events"])
        if turn["answer"]:
            render_markdown_with_images(turn["answer"])
        else:
            st.warning(f"The run ended without an answer ({turn['status']}).")
        if turn["sources"]:
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for source in turn["sources"]:
                    st.markdown(f"- {source}")
        st.caption(f"run `{turn['thread_id']}`")


# --- following a live run ---------------------------------------------------


def follow(active: dict) -> None:
    """Stream the active run's events into the page until it ends, then store the turn."""
    with st.chat_message("user"):
        st.markdown(active["goal"])
    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=True)
        with status:
            render_progress(active["events"])
        answer_box = st.empty()
        draft, final, done = "", None, None

        url = f"{API}/runs/{active['thread_id']}/events"
        with httpx.Client(timeout=httpx.Timeout(10, read=None)) as client:
            params = {"after": active["events_seen"]}
            with connect_sse(client, "GET", url, params=params) as source:
                for sse in source.iter_sse():
                    if not sse.data or not sse.id:  # keep-alive pings while the model thinks
                        continue
                    event = json.loads(sse.data)
                    active["events_seen"] = int(sse.id) + 1  # resume point if we reconnect
                    if event["type"] == "token":
                        draft += event["text"]
                        answer_box.markdown(draft + "▌")
                        continue
                    active["events"].append(event)
                    if event["type"] == "final":
                        final = event
                    elif event["type"] == "done":
                        done = event
                    else:
                        with status:
                            render_progress([event])

        label = {"completed": "Done", "failed": "Failed", "interrupted": "Waiting for you"}
        status.update(
            label=label.get(done["status"], "Stopped") if done else "Stopped",
            state="error" if done and done["status"] == "failed" else "complete",
            expanded=False,
        )

    if done and done["status"] == "interrupted":
        # The action agent is waiting for approval: keep the run active and ask the user.
        requests = [e["value"] for e in active["events"] if e["type"] == "interrupt"]
        active["pending"] = requests[-1] if requests else None
        active["draft"] = draft
        st.rerun()

    state.history.append(
        {
            "thread_id": active["thread_id"],
            "goal": active["goal"],
            "events": active["events"],
            "answer": final["answer"] if final else draft,
            "sources": final["sources"] if final else [],
            "status": done["status"] if done else "unknown",
        }
    )
    state.active = None
    st.rerun()


def ask_for_approval(active: dict) -> None:
    """Show the pending side effects and let the user approve or reject them."""
    request = active["pending"]
    with st.chat_message("user"):
        st.markdown(active["goal"])
    with st.chat_message("assistant"):
        with st.expander("How I did it", expanded=False):
            render_progress(active["events"])
        if active.get("draft"):
            render_markdown_with_images(active["draft"])
        st.info("**I'd like to make these changes. Nothing happens until you approve.**", icon="✋")
        for action in request["actions"]:
            warning = " — ⚠️ **overwrites an existing file**" if action["overwrite"] else ""
            st.markdown(
                f"`{action['tool']}` → `{action['path']}` ({action['chars']:,} chars){warning}"
            )
            with st.expander("Preview"):
                st.code(action["preview"], language="markdown")
        note = st.text_input("Note (optional)", key=f"note-{active['thread_id']}")
        approve, reject = st.columns(2)
        decision = None
        if approve.button("Approve", type="primary", use_container_width=True):
            decision = True
        if reject.button("Reject", use_container_width=True):
            decision = False
        if decision is not None:
            body = {"approved": decision, "note": note}
            api("POST", f"/runs/{active['thread_id']}/approval", json=body)
            active["pending"] = None
            st.rerun()


# --- page -------------------------------------------------------------------

with st.sidebar:
    st.title("🎼 OrchestraOS")
    st.caption("Local multi-agent assistant. Everything runs on your machine.")
    try:
        health = api("GET", "/health").json()
        if health["status"] == "ok":
            st.success("Models ready", icon="✅")
        elif not health["ollama"]["reachable"]:
            st.error("Ollama isn't running", icon="⚠️")
        else:
            missing = ", ".join(health["ollama"]["missing_models"])
            st.warning(f"Missing models: {missing}", icon="⚠️")
    except httpx.HTTPError:
        st.error(f"Can't reach the API at {API}", icon="⚠️")
        st.stop()

    st.subheader("Workspace")
    upload = st.file_uploader("Add a file (CSV, PDF, text)", key="upload")
    if upload is not None and state.get("uploaded") != upload.file_id:
        api("POST", "/files", files={"upload": (upload.name, upload.getvalue())})
        state.uploaded = upload.file_id
        st.toast(f"Added {upload.name}")
    files = [f for f in api("GET", "/files").json() if not f.startswith("outputs/")]
    st.markdown("\n".join(f"- `{f}`" for f in files) or "_empty_")

    if state.history and st.button("New conversation"):
        state.history = []
        st.rerun()

for turn in state.history:
    render_turn(turn)

if state.active and state.active.get("pending"):
    ask_for_approval(state.active)
elif state.active:
    follow(state.active)

if goal := st.chat_input("What should I do? e.g. “Analyse sales.csv and chart revenue”"):
    thread_id = api("POST", "/runs", json={"goal": goal}).json()["thread_id"]
    state.active = {"thread_id": thread_id, "goal": goal, "events_seen": 0, "events": []}
    st.rerun()
