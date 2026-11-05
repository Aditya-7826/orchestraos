"""Smoke test against the real local Ollama. Run with: uv run pytest -m live

Skipped by default (the `live` marker is deselected in pyproject.toml) and also
skipped if Ollama isn't reachable or the configured models aren't pulled.
"""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from orchestraos.config import Settings
from orchestraos.llm.router import ModelRouter, TaskKind
from orchestraos.ollama import OllamaUnavailable, list_local_models, missing_models

pytestmark = pytest.mark.live


class Route(BaseModel):
    agent: str


@pytest.fixture(scope="module")
def router() -> ModelRouter:
    settings = Settings()
    try:
        available = list_local_models(settings.ollama_base_url)
    except OllamaUnavailable:
        pytest.skip("Ollama not reachable")
    if missing := missing_models([settings.small_model, settings.large_model], available):
        pytest.skip(f"models not pulled: {missing}")
    return ModelRouter.from_settings(settings)


async def test_small_model_returns_routing_json(router):
    messages = [
        SystemMessage(
            "Pick the agent for the task. Agents: research, writing, analytics. "
            'Reply with JSON only, like {"agent": "research"}.'
        ),
        HumanMessage("Find recent articles comparing LangGraph and CrewAI."),
    ]
    result = await router.ainvoke_json(TaskKind.ROUTE, messages, Route)
    assert result.parsed.agent in {"research", "writing", "analytics"}


async def test_large_model_writes(router):
    result = await router.ainvoke(
        TaskKind.WRITE, [HumanMessage("Say hello in five words or fewer.")]
    )
    assert result.text.strip()
