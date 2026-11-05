import hashlib
import json
import math
import re

import pytest
from fakes import fake_router, system_prompt

from orchestraos.memory.agent import FACTS_SYSTEM, MemoryAgent, compose_context
from orchestraos.memory.graph import Fact, NullGraph
from orchestraos.memory.vector import MemoryHit, VectorMemory

DIMS = 256


class BagOfWordsEmbedder:
    """Deterministic stand-in for nomic-embed-text: texts sharing words get similar vectors."""

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * DIMS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def embed_documents(self, texts):
        return [self.embed(t) for t in texts]

    async def embed_query(self, text):
        return self.embed(text)


class FakeGraph(NullGraph):
    def __init__(self):
        self.facts: list[Fact] = []

    async def add_facts(self, facts):
        self.facts += [f for f in facts if f not in self.facts]

    async def related(self, text, limit=10):
        text = text.lower()
        return [f for f in self.facts if f.subject == "user" or f.subject in text][:limit]


def fact_llm(facts: list[dict], calls: list | None = None):
    def respond(messages):
        assert system_prompt(messages) == FACTS_SYSTEM
        if calls is not None:
            calls.append(str(messages[-1].content))
        return json.dumps({"facts": facts})

    return respond


@pytest.fixture
def vector(tmp_path):
    return VectorMemory(tmp_path / "chroma", BagOfWordsEmbedder())


async def test_past_work_is_recalled_for_a_similar_goal(vector):
    memory = MemoryAgent(fake_router(fact_llm([])), vector, FakeGraph())
    await memory.remember("compare langgraph and crewai frameworks", "LangGraph is lower level.")

    context = await memory.recall("compare the langgraph and crewai frameworks")
    assert "Related past work:" in context
    assert "LangGraph is lower level." in context


async def test_unrelated_goal_recalls_nothing(vector):
    memory = MemoryAgent(fake_router(fact_llm([])), vector, FakeGraph())
    await memory.remember("compare langgraph and crewai frameworks", "answer")
    assert await memory.recall("bake sourdough bread at home") == ""


async def test_user_facts_are_extracted_stored_and_recalled(vector):
    graph, calls = FakeGraph(), []
    facts = [{"subject": "User", "relation": "prefers", "object": "Bullet  Points"}]
    memory = MemoryAgent(fake_router(fact_llm(facts, calls)), vector, graph)

    await memory.remember("I prefer bullet points. Summarise LangGraph.", "...")
    assert calls == ["I prefer bullet points. Summarise LangGraph."]  # goal only, not the answer
    assert graph.facts == [Fact(subject="user", relation="prefers", object="bullet points")]

    context = await memory.recall("explain vector databases")  # a different, later session
    assert "Known about the user:\n- user prefers bullet points" in context


async def test_no_model_call_when_the_user_says_nothing_about_themselves(vector):
    calls = []
    memory = MemoryAgent(fake_router(fact_llm([], calls)), vector, FakeGraph())
    await memory.remember("Summarise the history of the printing press", "...")
    assert calls == []


async def test_remembering_the_same_fact_twice_keeps_one_copy(vector):
    facts = [{"subject": "user", "relation": "uses", "object": "fastapi"}]
    memory = MemoryAgent(fake_router(fact_llm(facts)), vector, NullGraph())
    await memory.remember("we use FastAPI", "a")
    await memory.remember("we use FastAPI", "a")
    hits = await vector.search("fastapi", kind="fact", k=10)
    assert [h.text for h in hits] == ["user uses fastapi"]


async def test_bad_extraction_output_stores_no_facts(vector):
    memory = MemoryAgent(fake_router(lambda messages: "not json"), vector, FakeGraph())
    await memory.remember("I like tables", "answer")
    assert await vector.search("tables", kind="fact") == []
    assert await vector.search("tables", kind="episode")  # the episode is still stored


async def test_memory_failures_never_break_a_run(vector):
    class BrokenGraph(NullGraph):
        async def related(self, text, limit=10):
            raise ConnectionError("neo4j down")

        async def add_facts(self, facts):
            raise ConnectionError("neo4j down")

    memory = MemoryAgent(fake_router(fact_llm([])), vector, BrokenGraph())
    assert await memory.recall("anything") == ""
    await memory.remember("I like tables", "answer")  # logs, doesn't raise


async def test_empty_store_recalls_nothing(vector):
    memory = MemoryAgent(fake_router(fact_llm([])), vector, NullGraph())
    assert await memory.recall("anything") == ""


def test_fact_normalization():
    fact = Fact(subject="  User ", relation="PREFERS", object="Dark   Mode")
    assert fact.text() == "user prefers dark mode"


def test_compose_context_formats_both_sections():
    episode = MemoryHit(
        text="Goal: x\nAnswer: y", kind="episode", distance=0.1, created="2026-11-05"
    )
    text = compose_context(["user prefers tables"], [episode])
    assert text == (
        "Known about the user:\n- user prefers tables\n\n"
        "Related past work:\n- (2026-11-05) Goal: x\nAnswer: y"
    )


# --- real Neo4j (optional) --------------------------------------------------


async def test_neo4j_round_trip():
    """Runs only when Neo4j is up (make up-graph) and NEO4J_URI is set in .env."""
    from orchestraos.config import Settings
    from orchestraos.memory.graph import Neo4jGraph

    settings = Settings()
    if not (settings.neo4j_uri and settings.neo4j_password):
        pytest.skip("NEO4J_URI not configured")
    try:
        graph = await Neo4jGraph.connect(
            settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
        )
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable: {exc}")
    try:
        fact = Fact(subject="user", relation="prefers", object="test-fact-zebra")
        await graph.add_facts([fact, fact])  # MERGE: stored once
        related = await graph.related("anything at all")
        assert related.count(fact) == 1
        assert fact in await graph.related("tell me about test-fact-zebra")
    finally:
        await graph._driver.execute_query(
            "MATCH (o:Entity {name: 'test-fact-zebra'}) DETACH DELETE o"
        )
        await graph.close()
