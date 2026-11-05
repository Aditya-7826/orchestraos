"""Turns LangGraph stream items into the small JSON events the UI understands.

The graph is streamed with stream_mode=["updates", "messages"] and subgraphs=True,
which yields (namespace, mode, chunk) tuples:
  * ((), "updates", {node: update})           a supervisor node finished
  * (("writing:<id>",), "messages", (msg, meta)) a token from the writing agent's LLM
Everything else (inner subgraph updates, tokens from other agents) is dropped.
"""

from typing import Any

from orchestraos.graph.state import AGENT_NAMES

PREVIEW_CHARS = 600

Event = dict[str, Any]


def to_events(namespace: tuple[str, ...], mode: str, chunk: Any) -> list[Event]:
    if mode == "messages":
        return token_events(namespace, chunk)
    if mode == "updates" and not namespace:
        return [event for node, update in chunk.items() for event in update_events(node, update)]
    return []


def token_events(namespace: tuple[str, ...], chunk: Any) -> list[Event]:
    # Only the writing agent's tokens are the user-facing answer; stream just those.
    if not namespace or not namespace[0].startswith("writing:"):
        return []
    message, _metadata = chunk
    text = message.content if isinstance(message.content, str) else ""
    return [{"type": "token", "text": text}] if text else []


def update_events(node: str, update: Any) -> list[Event]:
    if node == "__interrupt__":
        return [{"type": "interrupt", "value": interrupt.value} for interrupt in update]
    if not isinstance(update, dict):
        return []
    if node == "triage":
        return [{"type": "triage", "mode": update["mode"]}]
    if node == "plan":
        return [{"type": "plan", "steps": update["plan"]}]
    if node in AGENT_NAMES and update.get("results"):
        result = update["results"][-1]
        return [
            {
                "type": "step",
                "step_id": result["step_id"],
                "agent": result["agent"],
                "ok": result["ok"],
                "preview": result["output"][:PREVIEW_CHARS],
                "sources": result["sources"],
                "artifacts": result["artifacts"],
            }
        ]
    if node == "finalize":
        return [
            {
                "type": "final",
                "answer": update["final_answer"],
                "sources": update.get("sources", []),
                "artifacts": update.get("artifacts", []),
            }
        ]
    return []
