"""Memory agent: what the supervisor calls before planning and after answering.

    recall:   START -> search (vector episodes + vector facts + graph facts) -> compose -> END
    remember: START -> extract_facts (small model) -> store (vector + graph) -> END

Only things the *user* said become facts: facts are extracted from the goal,
never from model answers, so a hallucination can't turn into a "memory".
Memory is best-effort: if a store is down, runs carry on without it.
"""

import asyncio
import hashlib
import logging
import re
from datetime import UTC, datetime
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from orchestraos.llm.router import ModelRouter, ModelRouterError, TaskKind
from orchestraos.memory.graph import Fact, GraphStore
from orchestraos.memory.vector import MemoryHit, VectorMemory

logger = logging.getLogger(__name__)

MAX_EPISODES = 3
MAX_FACTS = 8
# Cosine-distance cut-offs, calibrated on nomic-embed-text: a related past run scored
# 0.13-0.16 and unrelated ones 0.49-0.58; a relevant user fact 0.45, unrelated ones 0.55-0.65.
EPISODE_MAX_DISTANCE = 0.35
FACT_MAX_DISTANCE = 0.55
EPISODE_ANSWER_CHARS = 600

# Cheap pre-check so we only spend a model call when the user talks about themselves.
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|my|me|we|we're|our|us|prefer|always|never)\b", re.I)

FACTS_SYSTEM = """Extract lasting facts the USER states about themselves, their preferences,
or their projects. Ignore the task itself and general knowledge.
Use short lowercase phrases. Use "user" as the subject for the user themself.

Examples:
"I prefer bullet points" -> {"subject": "user", "relation": "prefers", "object": "bullet points"}
"our app uses FastAPI" -> {"subject": "user's app", "relation": "uses", "object": "fastapi"}

Reply with JSON only: {"facts": [...]} (an empty list if there are none)."""


class Facts(BaseModel):
    facts: list[Fact] = Field(max_length=10)


class RecallState(TypedDict, total=False):
    goal: str
    episodes: list[MemoryHit]
    facts: list[str]
    context: str


class RememberState(TypedDict, total=False):
    goal: str
    answer: str
    facts: list[Fact]


def doc_id(prefix: str, text: str) -> str:
    return f"{prefix}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"


def compose_context(facts: list[str], episodes: list[MemoryHit]) -> str:
    parts = []
    if facts:
        parts.append("Known about the user:\n" + "\n".join(f"- {f}" for f in facts))
    if episodes:
        lines = [f"- ({e.created}) {e.text}" for e in episodes]
        parts.append("Related past work:\n" + "\n".join(lines))
    return "\n\n".join(parts)


class MemoryAgent:
    def __init__(self, router: ModelRouter, vector: VectorMemory, graph: GraphStore):
        self.router, self.vector, self.graph = router, vector, graph
        self._recall = self._build_recall()
        self._remember = self._build_remember()

    # --- the Memory interface the supervisor uses ---------------------------

    async def recall(self, goal: str) -> str:
        try:
            return (await self._recall.ainvoke({"goal": goal}))["context"]
        except Exception:
            logger.exception("memory recall failed; continuing without memory")
            return ""

    async def remember(self, goal: str, answer: str) -> None:
        try:
            await self._remember.ainvoke({"goal": goal, "answer": answer})
        except Exception:
            logger.exception("memory store failed; this run won't be remembered")

    # --- subgraphs -----------------------------------------------------------

    def _build_recall(self):
        async def search(state: RecallState) -> RecallState:
            episodes, fact_hits, graph_facts = await asyncio.gather(
                self.vector.search(state["goal"], kind="episode", k=MAX_EPISODES),
                self.vector.search(state["goal"], kind="fact", k=MAX_FACTS),
                self.graph.related(state["goal"], limit=MAX_FACTS),
            )
            facts = [f.text() for f in graph_facts]
            facts += [h.text for h in fact_hits if h.distance <= FACT_MAX_DISTANCE]
            return {
                "episodes": [e for e in episodes if e.distance <= EPISODE_MAX_DISTANCE],
                "facts": list(dict.fromkeys(facts))[:MAX_FACTS],  # de-duplicated, in order
            }

        def compose(state: RecallState) -> RecallState:
            return {"context": compose_context(state["facts"], state["episodes"])}

        graph = StateGraph(RecallState)
        graph.add_node("search", search)
        graph.add_node("compose", compose)
        graph.add_edge(START, "search")
        graph.add_edge("search", "compose")
        graph.add_edge("compose", END)
        return graph.compile(name="memory_recall")

    def _build_remember(self):
        async def extract_facts(state: RememberState) -> RememberState:
            if not _FIRST_PERSON.search(state["goal"]):
                return {"facts": []}
            messages = [SystemMessage(FACTS_SYSTEM), HumanMessage(state["goal"])]
            try:
                result = await self.router.ainvoke_json(TaskKind.EXTRACT, messages, Facts)
            except ModelRouterError:
                return {"facts": []}
            return {"facts": result.parsed.facts}

        async def store(state: RememberState) -> RememberState:
            today = datetime.now(UTC).date().isoformat()
            episode = f"Goal: {state['goal']}\nAnswer: {state['answer'][:EPISODE_ANSWER_CHARS]}"
            fact_texts = [f.text() for f in state["facts"]]
            await asyncio.gather(
                self.vector.add([doc_id("episode", episode)], [episode], "episode", today),
                self.vector.add([doc_id("fact", t) for t in fact_texts], fact_texts, "fact", today),
                self.graph.add_facts(state["facts"]),
            )
            return {}

        graph = StateGraph(RememberState)
        graph.add_node("extract_facts", extract_facts)
        graph.add_node("store", store)
        graph.add_edge(START, "extract_facts")
        graph.add_edge("extract_facts", "store")
        graph.add_edge("store", END)
        return graph.compile(name="memory_remember")
