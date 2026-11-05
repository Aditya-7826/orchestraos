"""Research agent: web search + local documents -> findings with numbered sources.

    START -> plan_queries -> gather -> synthesize -> END

A fixed pipeline rather than a free-form tool-calling loop: 7B local models are
much more reliable when each LLM call has one narrow job (write queries, then
write findings) and the tool calls themselves are made by code.
"""

import asyncio
import logging
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from orchestraos.agents.base import Agent, AgentState
from orchestraos.agents.common import mentioned_files, task_text
from orchestraos.agents.prompts import QUERY_SYSTEM, SYNTHESIZE_SYSTEM
from orchestraos.llm.router import ModelRouter, ModelRouterError, TaskKind
from orchestraos.tools.client import ToolCaller, ToolCallError

logger = logging.getLogger(__name__)

RESULTS_PER_QUERY = 4
MAX_WEB_PAGES = 3
MAX_LOCAL_DOCS = 3
PAGE_CHARS = 2500
DOC_CHARS = 4000
DOC_SUFFIXES = (".md", ".txt", ".pdf", ".html", ".json")


class Source(TypedDict):
    ref: str  # URL or workspace path
    title: str
    text: str


class ResearchState(AgentState, total=False):
    queries: list[str]
    collected: list[Source]


class Queries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3)


def interleave(hit_lists: list[list[dict]], limit: int) -> list[dict]:
    """Top hit of each query first, then the second of each, ... without duplicate URLs."""
    picked, seen = [], set()
    for rank in range(max((len(h) for h in hit_lists), default=0)):
        for hits in hit_lists:
            if rank < len(hits) and hits[rank]["url"] not in seen:
                seen.add(hits[rank]["url"])
                picked.append(hits[rank])
    return picked[:limit]


def numbered_sources(sources: list[Source]) -> str:
    return "\n\n".join(
        f"[{i}] {s['title'] or s['ref']} ({s['ref']})\n{s['text']}"
        for i, s in enumerate(sources, 1)
    )


def build_research_agent(router: ModelRouter, tools: ToolCaller) -> Agent:
    async def plan_queries(state: ResearchState) -> ResearchState:
        messages = [SystemMessage(QUERY_SYSTEM), HumanMessage(state["instruction"])]
        try:
            result = await router.ainvoke_json(TaskKind.PLAN, messages, Queries)
            queries = result.parsed.queries
        except ModelRouterError:
            queries = [state["instruction"][:200]]  # the instruction itself is a usable query
        return {"queries": queries}

    async def search_web(queries: list[str]) -> list[Source]:
        async def search(query: str) -> list[dict]:
            try:
                return await tools.call("web_search", query=query, max_results=RESULTS_PER_QUERY)
            except ToolCallError as exc:
                logger.warning("search failed for %r: %s", query, exc)
                return []

        hits = interleave(await asyncio.gather(*(search(q) for q in queries)), MAX_WEB_PAGES)

        async def read(hit: dict) -> Source:
            try:
                page = await tools.call("fetch_page", url=hit["url"], max_chars=PAGE_CHARS)
                text = page["text"] or hit["snippet"]
            except ToolCallError:
                text = hit["snippet"]  # page blocked or unreachable: the snippet still helps
            return Source(ref=hit["url"], title=hit["title"], text=text)

        return list(await asyncio.gather(*(read(h) for h in hits)))

    async def read_local_docs(state: ResearchState) -> list[Source]:
        try:
            files = await tools.call("list_files")
        except ToolCallError:
            return []
        docs = [f for f in files if f.endswith(DOC_SUFFIXES) and not f.startswith("outputs/")]
        sources = []
        for path in mentioned_files(docs, task_text(state))[:MAX_LOCAL_DOCS]:
            try:
                text = await tools.call("read_file", path=path, max_chars=DOC_CHARS)
            except ToolCallError:
                continue
            sources.append(Source(ref=path, title=path, text=text))
        return sources

    async def gather(state: ResearchState) -> ResearchState:
        local, web = await asyncio.gather(read_local_docs(state), search_web(state["queries"]))
        return {"collected": [*local, *web]}

    async def synthesize(state: ResearchState) -> ResearchState:
        sources = state["collected"]
        if not sources:
            return {"output": "No sources found for this task.", "sources": [], "artifacts": []}

        prompt = f"Task: {state['instruction']}\n\nSources:\n\n{numbered_sources(sources)}"
        messages = [SystemMessage(SYNTHESIZE_SYSTEM), HumanMessage(prompt)]
        findings = (await router.ainvoke(TaskKind.WRITE, messages)).text.strip()
        reference_list = "\n".join(f"[{i}] {s['ref']}" for i, s in enumerate(sources, 1))
        return {
            "output": f"{findings}\n\nSources:\n{reference_list}",
            "sources": [s["ref"] for s in sources],
            "artifacts": [],
        }

    graph = StateGraph(ResearchState, input_schema=AgentState, output_schema=AgentState)
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("gather", gather)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "plan_queries")
    graph.add_edge("plan_queries", "gather")
    graph.add_edge("gather", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile(name="research_agent")
