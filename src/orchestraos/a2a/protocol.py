"""How a research task and its result are put into A2A messages (shared by server and client).

Request:  one user Message with a data part {"instruction", "goal", "context"}
          (a plain-text message also works: its text is the instruction).
Result:   one Artifact named "findings" with a text part (the findings, markdown)
          and a data part {"sources": [...]}.
"""

import uuid
from typing import Any

from a2a.helpers import get_data_parts, get_text_parts, new_data_part, new_text_part
from a2a.types import Artifact, Message, Part, Role

ARTIFACT_NAME = "findings"


def task_message(instruction: str, goal: str = "", context: str = "") -> Message:
    data = {"instruction": instruction, "goal": goal, "context": context}
    return Message(role=Role.ROLE_USER, message_id=uuid.uuid4().hex, parts=[new_data_part(data)])


def read_task(message: Message) -> dict[str, str]:
    """The research task in `message`; tolerant of plain-text messages from other clients."""
    for data in get_data_parts(message.parts):
        if isinstance(data, dict) and data.get("instruction"):
            return {k: str(data.get(k, "")) for k in ("instruction", "goal", "context")}
    text = "\n".join(get_text_parts(message.parts)).strip()
    if not text:
        raise ValueError("the message has no research instruction")
    return {"instruction": text, "goal": text, "context": ""}


def result_parts(findings: str, sources: list[str]) -> list[Part]:
    return [new_text_part(findings), new_data_part({"sources": sources})]


def read_result(artifacts: list[Artifact]) -> dict[str, Any]:
    """{"output": findings, "sources": [...]} from a completed task's artifacts."""
    for artifact in artifacts:
        if artifact.name != ARTIFACT_NAME:
            continue
        sources: list[str] = []
        for data in get_data_parts(artifact.parts):
            if isinstance(data, dict):
                sources += [str(s) for s in data.get("sources", [])]
        return {"output": "\n".join(get_text_parts(artifact.parts)), "sources": sources}
    raise ValueError("the remote agent returned no findings")
