import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from orchestraos.llm.router import (
    InvalidJSONError,
    ModelRouter,
    ModelRouterError,
    NamedModel,
    TaskKind,
    parse_json_output,
)

MESSAGES = [HumanMessage("hi")]


class ScriptedModel:
    """Returns (or raises) the scripted outputs in order and records each call."""

    def __init__(self, *outputs: str | Exception):
        self.outputs = list(outputs)
        self.calls = 0

    async def ainvoke(self, input: list[BaseMessage]) -> AIMessage:
        self.calls += 1
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return AIMessage(output)


class Route(BaseModel):
    agent: str


def make_router(small: ScriptedModel, large: ScriptedModel) -> ModelRouter:
    return ModelRouter(small=NamedModel("small", small), large=NamedModel("large", large))


# --- model choice -----------------------------------------------------------


@pytest.mark.parametrize("kind", [TaskKind.ROUTE, TaskKind.CLASSIFY])
async def test_routing_tasks_use_small_model(kind):
    small, large = ScriptedModel("ok"), ScriptedModel()
    result = await make_router(small, large).ainvoke(kind, MESSAGES)
    assert result.model == "small" and large.calls == 0


@pytest.mark.parametrize("kind", [TaskKind.PLAN, TaskKind.WRITE])
async def test_planning_tasks_use_large_model(kind):
    small, large = ScriptedModel(), ScriptedModel("ok")
    result = await make_router(small, large).ainvoke(kind, MESSAGES)
    assert result.model == "large" and small.calls == 0


# --- retry and fallback -----------------------------------------------------


async def test_retries_primary_once_before_falling_back():
    small, large = ScriptedModel(), ScriptedModel(RuntimeError("timeout"), "second try")
    result = await make_router(small, large).ainvoke(TaskKind.PLAN, MESSAGES)
    assert result.text == "second try"
    assert result.model == "large" and not result.used_fallback
    assert small.calls == 0


async def test_falls_back_after_two_primary_failures():
    small = ScriptedModel("rescued")
    large = ScriptedModel(RuntimeError("a"), RuntimeError("b"))
    result = await make_router(small, large).ainvoke(TaskKind.WRITE, MESSAGES)
    assert result.model == "small" and result.used_fallback
    assert [a.model for a in result.attempts] == ["large", "large", "small"]


async def test_raises_when_every_attempt_fails():
    small = ScriptedModel(RuntimeError("s"))
    large = ScriptedModel(RuntimeError("l1"), RuntimeError("l2"))
    with pytest.raises(ModelRouterError) as info:
        await make_router(small, large).ainvoke(TaskKind.PLAN, MESSAGES)
    assert len(info.value.attempts) == 3 and not any(a.ok for a in info.value.attempts)


async def test_invalid_json_counts_as_failure():
    small = ScriptedModel("not json", '{"agent": "research"}')
    large = ScriptedModel()
    result = await make_router(small, large).ainvoke_json(TaskKind.ROUTE, MESSAGES, Route)
    assert result.parsed == Route(agent="research")
    assert [a.ok for a in result.attempts] == [False, True]


async def test_wrong_schema_falls_back_to_other_model():
    small = ScriptedModel('{"nope": 1}', '{"nope": 2}')
    large = ScriptedModel('{"agent": "writing"}')
    result = await make_router(small, large).ainvoke_json(TaskKind.CLASSIFY, MESSAGES, Route)
    assert result.model == "large" and result.parsed == Route(agent="writing")


# --- JSON parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"agent": "research"}',
        '```json\n{"agent": "research"}\n```',
        'Sure! Here you go: {"agent": "research"} Hope that helps.',
    ],
)
def test_parse_json_tolerates_common_wrapping(text):
    assert parse_json_output(text, Route) == Route(agent="research")


@pytest.mark.parametrize("text", ["", "no braces here", '{"agent": }', '{"other": "x"}'])
def test_parse_json_rejects_bad_output(text):
    with pytest.raises(InvalidJSONError):
        parse_json_output(text, Route)
