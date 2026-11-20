"""Small helpers shared by the agents."""

import re

from langgraph.config import get_stream_writer

from orchestraos.agents.base import AgentState


class AgentError(RuntimeError):
    """An agent couldn't do its step. The supervisor records it as a failed step."""


def report_progress(text: str) -> None:
    """Tell the UI, in one line, what this agent is doing right now.

    Sent on LangGraph's "custom" stream. Outside a streamed run (CLI, tests) it does nothing.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:  # not inside a graph run
        return
    writer({"type": "activity", "text": text})


def task_text(state: AgentState) -> str:
    """The step's instruction plus the overall goal, for matching file names etc."""
    return f"{state.get('instruction', '')}\n{state.get('goal', '')}"


def mentioned_files(files: list[str], text: str) -> list[str]:
    """Files whose name (with or without extension) appears in `text`."""
    lowered = text.lower()
    found = []
    for path in files:
        name = path.rsplit("/", 1)[-1].lower()
        stem = name.rsplit(".", 1)[0]
        if name in lowered or re.search(rf"\b{re.escape(stem)}\b", lowered):
            found.append(path)
    return found


_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """The code inside the first ```python block, or the whole reply if there's no block."""
    match = _CODE_BLOCK.search(text)
    return (match.group(1) if match else text).strip()
