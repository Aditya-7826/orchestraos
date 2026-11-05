import json

import pytest
from fakes import FakeTools, fake_router, system_prompt

from orchestraos.agents.analytics import MAX_FIXES, build_analytics_agent
from orchestraos.agents.common import AgentError, extract_code, mentioned_files
from orchestraos.agents.prompts import CODE_SYSTEM, FIX_CODE_SYSTEM, QUERY_SYSTEM
from orchestraos.agents.research import build_research_agent, interleave
from orchestraos.agents.writing import build_writing_agent
from orchestraos.tools.client import ToolCallError

TASK = {"goal": "goal", "instruction": "Compare LangGraph and CrewAI", "context": "", "memory": ""}


def raises(exc: Exception):
    def handler(**_):
        raise exc

    return handler


# --- research ---------------------------------------------------------------


def research_llm(queries=("langgraph", "crewai")):
    prompts = []

    def respond(messages):
        prompt = system_prompt(messages)
        if prompt == QUERY_SYSTEM:
            return json.dumps({"queries": list(queries)})
        prompts.append(str(messages[-1].content))
        return "LangGraph is a graph framework [1]. CrewAI uses roles [2]."

    return respond, prompts


def hit(n: int) -> dict:
    return {"title": f"Title {n}", "url": f"https://site{n}.example", "snippet": f"snippet {n}"}


async def test_research_searches_reads_and_cites():
    respond, prompts = research_llm()
    tools = FakeTools(
        list_files=lambda **_: [],
        web_search=lambda query, **_: [hit(1), hit(2)] if query == "langgraph" else [hit(3)],
        fetch_page=lambda url, **_: {"url": url, "title": "", "text": f"page text of {url}"},
    )
    out = await build_research_agent(fake_router(respond), tools).ainvoke(TASK)

    # Top hit of each query first, capped at 3 pages.
    assert out["sources"] == [
        "https://site1.example",
        "https://site3.example",
        "https://site2.example",
    ]
    assert "page text of https://site1.example" in prompts[0]
    assert "[1]" in out["output"] and "Sources:\n[1] https://site1.example" in out["output"]


async def test_research_uses_snippet_when_page_fetch_fails():
    respond, prompts = research_llm(queries=("q",))
    tools = FakeTools(
        list_files=lambda **_: [],
        web_search=lambda **_: [hit(1)],
        fetch_page=raises(ToolCallError("blocked")),
    )
    out = await build_research_agent(fake_router(respond), tools).ainvoke(TASK)
    assert out["sources"] == ["https://site1.example"]
    assert "snippet 1" in prompts[0]


async def test_research_reads_local_docs_named_in_the_task():
    respond, prompts = research_llm(queries=("q",))
    tools = FakeTools(
        list_files=lambda **_: ["notes.md", "other.txt", "data.csv", "outputs/old/report.md"],
        read_file=lambda path, **_: f"contents of {path}",
        web_search=lambda **_: [],
    )
    task = {**TASK, "instruction": "Summarise notes.md and compare with the web"}
    out = await build_research_agent(fake_router(respond), tools).ainvoke(task)
    assert out["sources"] == ["notes.md"]
    assert [c["path"] for c in tools.called("read_file")] == ["notes.md"]


async def test_research_with_nothing_found_says_so():
    respond, _ = research_llm(queries=("q",))
    tools = FakeTools(list_files=lambda **_: [], web_search=raises(ToolCallError("rate limited")))
    out = await build_research_agent(fake_router(respond), tools).ainvoke(TASK)
    assert out["output"] == "No sources found for this task." and out["sources"] == []


async def test_research_falls_back_to_instruction_as_query():
    def respond(messages):
        return "not json" if system_prompt(messages) == QUERY_SYSTEM else "findings"

    tools = FakeTools(list_files=lambda **_: [], web_search=lambda **_: [])
    await build_research_agent(fake_router(respond), tools).ainvoke(TASK)
    assert tools.called("web_search")[0]["query"] == TASK["instruction"]


def test_interleave_dedupes_and_limits():
    a, b = [hit(1), hit(2)], [hit(1), hit(3)]
    assert [h["url"] for h in interleave([a, b], 10)] == [
        "https://site1.example", "https://site2.example", "https://site3.example",
    ]  # fmt: skip
    assert len(interleave([a, b], 2)) == 2


# --- analytics --------------------------------------------------------------


def sandbox_result(exit_code=0, stdout="", stderr="", files=()):
    return {"exit_code": exit_code, "timed_out": False, "stdout": stdout, "stderr": stderr,
            "files": list(files)}  # fmt: skip


PROFILE_OK = sandbox_result(stdout="== /data/sales.csv: 2 rows\nColumn types:\nmonth int64")


def analytics_tools(*runs, files=("sales.csv",), profile=PROFILE_OK):
    """Fake tools: the profiling run gets `profile`, model-written code gets `runs` in order."""
    queue = list(runs)

    def run_python(code, **_):
        return profile if "Column types" in code else queue.pop(0)

    return FakeTools(
        list_files=lambda **_: list(files),
        read_file=lambda path, **_: "month,sales\n1,10",
        run_python=run_python,
    )


def model_runs(tools: FakeTools) -> list[str]:
    """Code the model wrote that was run (excludes the fixed profiling script)."""
    return [c["code"] for c in tools.called("run_python") if "Column types" not in c["code"]]


def code_llm():
    def respond(messages):
        prompt = system_prompt(messages)
        if prompt == CODE_SYSTEM:
            return "Here:\n```python\nprint('first')\n```"
        assert prompt == FIX_CODE_SYSTEM
        return "```python\nprint('fixed')\n```"

    return respond


ANALYTICS_TASK = {**TASK, "instruction": "Total sales by month from sales.csv with a chart"}


async def test_analytics_runs_code_and_reports_charts():
    tools = analytics_tools(sandbox_result(stdout="total: 40\n", files=["outputs/x/chart.png"]))
    out = await build_analytics_agent(fake_router(code_llm()), tools).ainvoke(ANALYTICS_TASK)
    assert model_runs(tools) == ["print('first')"]
    assert "total: 40" in out["output"]
    assert out["artifacts"] == ["outputs/x/chart.png"]
    assert out["sources"] == ["sales.csv"]


async def test_analytics_feeds_errors_back_until_fixed():
    tools = analytics_tools(
        sandbox_result(exit_code=1, stderr="KeyError: 'Sales'"),
        sandbox_result(stdout="ok"),
    )
    llm = fake_router(code_llm())
    out = await build_analytics_agent(llm, tools).ainvoke(ANALYTICS_TASK)
    assert model_runs(tools) == ["print('first')", "print('fixed')"]
    fix_prompt = str(llm.large.model.calls[-1][-1].content)
    assert "KeyError: 'Sales'" in fix_prompt
    assert "ok" in out["output"]


async def test_analytics_gives_up_after_max_fixes():
    traceback = "Traceback (most recent call last):\n  File x\nValueError: boom\n"
    failures = [sandbox_result(exit_code=1, stderr=traceback) for _ in range(MAX_FIXES + 1)]
    tools = analytics_tools(*failures)
    with pytest.raises(AgentError, match="fixes: ValueError: boom") as info:
        await build_analytics_agent(fake_router(code_llm()), tools).ainvoke(ANALYTICS_TASK)
    assert "Traceback" not in str(info.value)  # only the useful last line
    assert len(model_runs(tools)) == MAX_FIXES + 1


async def test_analytics_shows_the_model_the_data_profile():
    tools = analytics_tools(sandbox_result())
    llm = fake_router(code_llm())
    await build_analytics_agent(llm, tools).ainvoke(ANALYTICS_TASK)
    assert "month int64" in str(llm.large.model.calls[0][-1].content)


async def test_analytics_falls_back_to_raw_preview_if_profiling_fails():
    tools = analytics_tools(sandbox_result(), profile=sandbox_result(exit_code=1, stderr="bad"))
    llm = fake_router(code_llm())
    await build_analytics_agent(llm, tools).ainvoke(ANALYTICS_TASK)
    assert "month,sales" in str(llm.large.model.calls[0][-1].content)


async def test_analytics_without_data_files_fails_clearly():
    tools = analytics_tools(files=("notes.md", "outputs/old/data.csv"))
    with pytest.raises(AgentError, match="no data files"):
        await build_analytics_agent(fake_router(code_llm()), tools).ainvoke(ANALYTICS_TASK)


async def test_analytics_uses_all_data_files_when_none_named():
    tools = analytics_tools(sandbox_result(), files=("a.csv", "b.csv"))
    task = {**TASK, "instruction": "Find trends in the data"}
    out = await build_analytics_agent(fake_router(code_llm()), tools).ainvoke(task)
    assert out["sources"] == ["a.csv", "b.csv"]


# --- writing ----------------------------------------------------------------


async def test_writing_includes_goal_context_and_memory():
    seen = []

    def respond(messages):
        seen.append(str(messages[-1].content))
        return "  # Report  "

    task = {**TASK, "context": "### Step 1 — research\nfacts [1]", "memory": "likes tables"}
    out = await build_writing_agent(fake_router(respond)).ainvoke(task)
    assert out["output"] == "# Report"
    assert "facts [1]" in seen[0] and "likes tables" in seen[0] and "goal" in seen[0]


# --- helpers ----------------------------------------------------------------


def test_mentioned_files_matches_name_or_stem():
    files = ["sales.csv", "reports/q1.pdf", "salesforce.csv"]
    found = mentioned_files(files, "analyse the SALES data and q1.pdf")
    assert found == ["sales.csv", "reports/q1.pdf"]


@pytest.mark.parametrize(
    "reply",
    ["```python\nx = 1\n```", "Sure!\n```py\nx = 1\n```\nDone.", "x = 1", "```\nx = 1\n```"],
)
def test_extract_code(reply):
    assert extract_code(reply) == "x = 1"
