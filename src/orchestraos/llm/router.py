"""Model router: picks the small or large model for each step, with retry and fallback.

Policy (from the brief):
  * routing / classification -> small model (~3B, fast)
  * planning / writing        -> large model (~7-8B, better reasoning + tool calling)
  * if a call fails or returns invalid JSON: retry the same model once,
    then make one attempt on the other model. If that fails too, raise.

The router only depends on the tiny `ChatModel` protocol below (anything with
`.ainvoke(messages)`), so tests use scripted fakes and never need Ollama.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from orchestraos.config import Settings

logger = logging.getLogger(__name__)


class TaskKind(StrEnum):
    ROUTE = "route"
    CLASSIFY = "classify"
    PLAN = "plan"
    WRITE = "write"
    CODE = "code"


SMALL_MODEL_TASKS = frozenset({TaskKind.ROUTE, TaskKind.CLASSIFY})


class ChatModel(Protocol):
    async def ainvoke(self, input: list[BaseMessage]) -> BaseMessage: ...


@dataclass(frozen=True)
class NamedModel:
    name: str
    model: ChatModel


@dataclass(frozen=True)
class Attempt:
    model: str
    ok: bool
    error: str | None = None


@dataclass
class RouterResult:
    text: str
    model: str  # which model produced the answer
    attempts: list[Attempt] = field(default_factory=list)
    parsed: BaseModel | None = None  # set by ainvoke_json

    @property
    def used_fallback(self) -> bool:
        return self.attempts[0].model != self.model


class ModelRouterError(Exception):
    def __init__(self, kind: TaskKind, attempts: list[Attempt]):
        self.attempts = attempts
        summary = "; ".join(f"{a.model}: {a.error}" for a in attempts)
        super().__init__(f"all models failed for {kind} task ({summary})")


class InvalidJSONError(ValueError):
    """The model answered, but not with JSON matching the expected schema."""


def message_text(message: BaseMessage) -> str:
    """Model output as plain text (content can be a string or a list of content blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else str(block.get("text", "")) for block in content
    )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_output(text: str, schema: type[BaseModel]) -> BaseModel:
    """Parse model output into `schema`, tolerating ```json fences and text around the object."""
    candidate = text.strip()
    if fenced := _FENCE.search(candidate):
        candidate = fenced.group(1)
    elif "{" in candidate:
        # Small models often add a sentence before/after; keep the outermost {...}.
        candidate = candidate[candidate.index("{") : candidate.rindex("}") + 1]

    try:
        return schema.model_validate(json.loads(candidate))
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise InvalidJSONError(f"invalid JSON for {schema.__name__}: {exc}") from exc


class ModelRouter:
    def __init__(self, small: NamedModel, large: NamedModel):
        self.small = small
        self.large = large

    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelRouter":
        # Imported here so the core router logic doesn't require langchain-ollama.
        from langchain_ollama import ChatOllama

        def ollama(name: str, num_ctx: int) -> NamedModel:
            model = ChatOllama(
                model=name, base_url=settings.ollama_base_url, temperature=0, num_ctx=num_ctx
            )
            return NamedModel(name, model)

        return cls(
            small=ollama(settings.small_model, settings.small_model_ctx),
            large=ollama(settings.large_model, settings.large_model_ctx),
        )

    def order_for(self, kind: TaskKind) -> tuple[NamedModel, NamedModel]:
        """(primary, fallback) for a task kind."""
        if kind in SMALL_MODEL_TASKS:
            return self.small, self.large
        return self.large, self.small

    async def ainvoke(self, kind: TaskKind, messages: list[BaseMessage]) -> RouterResult:
        return await self._run(kind, messages, schema=None)

    async def ainvoke_json(
        self, kind: TaskKind, messages: list[BaseMessage], schema: type[BaseModel]
    ) -> RouterResult:
        """Like `ainvoke`, but the answer must parse into `schema` (else it counts as a failure)."""
        return await self._run(kind, messages, schema=schema)

    async def _run(
        self, kind: TaskKind, messages: list[BaseMessage], schema: type[BaseModel] | None
    ) -> RouterResult:
        primary, fallback = self.order_for(kind)
        # Primary gets two tries (first call + one retry); fallback gets one.
        plan = [primary, primary, fallback]
        attempts: list[Attempt] = []

        for candidate in plan:
            try:
                text = message_text(await candidate.model.ainvoke(messages))
                parsed = parse_json_output(text, schema) if schema else None
            except Exception as exc:  # any model/transport/parse failure triggers the policy
                logger.warning("%s task failed on %s: %s", kind, candidate.name, exc)
                attempts.append(Attempt(candidate.name, ok=False, error=_short(exc)))
                continue

            attempts.append(Attempt(candidate.name, ok=True))
            return RouterResult(text=text, model=candidate.name, attempts=attempts, parsed=parsed)

        raise ModelRouterError(kind, attempts)


def _short(exc: Exception, limit: int = 200) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."
