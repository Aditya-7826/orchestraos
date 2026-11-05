"""Analytics agent: writes pandas/matplotlib code and runs it in the Docker sandbox.

    START -> find_data (+ profile) -> write_code -> run_code --ok--> report -> END
                                          ^   |
                                          |   +--error (max 2 fixes)--> fix_code
                                          +-----------------------------+

The model never touches the data directly: it sees a preview of each file,
writes a script, and the sandbox runs it with no network access. If the script
fails, the error goes back to the model to fix, up to MAX_FIXES times.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from orchestraos.agents.base import Agent, AgentState
from orchestraos.agents.common import AgentError, extract_code, mentioned_files, task_text
from orchestraos.agents.prompts import CODE_SYSTEM, FIX_CODE_SYSTEM
from orchestraos.llm.router import ModelRouter, TaskKind
from orchestraos.tools.client import ToolCaller

DATA_SUFFIXES = (".csv", ".tsv", ".json")
MAX_DATA_FILES = 3
PREVIEW_CHARS = 1500
MAX_FIXES = 2
RUN_TIMEOUT_SECONDS = 60


def profile_script(paths: list[str]) -> str:
    """A fixed (not model-written) script that prints each file's shape, dtypes and first rows."""
    return f"""
import pandas as pd
for path in {[f"/data/{p}" for p in paths]!r}:
    if path.endswith(".json"):
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path, sep="\\t" if path.endswith(".tsv") else ",")
    print(f"== {{path}}: {{len(df)}} rows x {{len(df.columns)}} columns")
    print("Column types:")
    print(df.dtypes.to_string())
    print("First rows:")
    print(df.head(5).to_string(index=False))
    print()
"""


def last_line(text: str) -> str:
    """The last non-empty line, e.g. 'KeyError: ...' at the end of a traceback."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "unknown error"


class AnalyticsState(AgentState, total=False):
    data_files: list[str]
    previews: str
    code: str
    run: dict  # last SandboxResult
    fixes: int


def build_analytics_agent(router: ModelRouter, tools: ToolCaller) -> Agent:
    async def find_data(state: AnalyticsState) -> AnalyticsState:
        files = await tools.call("list_files")
        data = [f for f in files if f.endswith(DATA_SUFFIXES) and not f.startswith("outputs/")]
        chosen = mentioned_files(data, task_text(state)) or data
        if not chosen:
            raise AgentError("no data files (.csv, .tsv, .json) found in the workspace folder")
        chosen = chosen[:MAX_DATA_FILES]

        # Profile the files in the sandbox so the model sees real column names and dtypes
        # (guessing types from raw CSV lines led to code that summed date columns).
        profile = await tools.call("run_python", code=profile_script(chosen), timeout_seconds=30)
        if profile["exit_code"] == 0:
            previews = profile["stdout"]
        else:  # e.g. a malformed file: fall back to showing the raw first lines
            heads = [
                f"/data/{path} (first lines):\n"
                + await tools.call("read_file", path=path, max_chars=PREVIEW_CHARS)
                for path in chosen
            ]
            previews = "\n\n".join(heads)
        return {"data_files": chosen, "previews": previews, "fixes": 0}

    async def write_code(state: AnalyticsState) -> AnalyticsState:
        prompt = f"Task: {state['instruction']}\n\nData files:\n{state['previews']}"
        messages = [SystemMessage(CODE_SYSTEM), HumanMessage(prompt)]
        reply = await router.ainvoke(TaskKind.CODE, messages)
        return {"code": extract_code(reply.text)}

    async def run_code(state: AnalyticsState) -> AnalyticsState:
        result = await tools.call(
            "run_python", code=state["code"], timeout_seconds=RUN_TIMEOUT_SECONDS
        )
        return {"run": result}

    def after_run(state: AnalyticsState) -> str:
        if state["run"]["exit_code"] == 0:
            return "report"
        return "fix_code" if state["fixes"] < MAX_FIXES else "give_up"

    async def fix_code(state: AnalyticsState) -> AnalyticsState:
        error = state["run"]["stderr"][-2000:]  # the traceback's end is the useful part
        prompt = (
            f"Task: {state['instruction']}\n\nData files:\n{state['previews']}\n\n"
            f"Script:\n```python\n{state['code']}\n```\n\nError:\n{error}"
        )
        messages = [SystemMessage(FIX_CODE_SYSTEM), HumanMessage(prompt)]
        reply = await router.ainvoke(TaskKind.CODE, messages)
        return {"code": extract_code(reply.text), "fixes": state["fixes"] + 1}

    def give_up(state: AnalyticsState) -> AnalyticsState:
        raise AgentError(
            f"analysis failed after {MAX_FIXES} fixes: {last_line(state['run']['stderr'])}"
        )

    def report(state: AnalyticsState) -> AnalyticsState:
        charts = state["run"]["files"]
        output = f"Analysed: {', '.join(state['data_files'])}\n\nResults:\n{state['run']['stdout']}"
        if charts:
            output += "\nCharts:\n" + "\n".join(f"- {path}" for path in charts)
        return {"output": output, "sources": state["data_files"], "artifacts": charts}

    graph = StateGraph(AnalyticsState, input_schema=AgentState, output_schema=AgentState)
    graph.add_node("find_data", find_data)
    graph.add_node("write_code", write_code)
    graph.add_node("run_code", run_code)
    graph.add_node("fix_code", fix_code)
    graph.add_node("give_up", give_up)
    graph.add_node("report", report)
    graph.add_edge(START, "find_data")
    graph.add_edge("find_data", "write_code")
    graph.add_edge("write_code", "run_code")
    graph.add_conditional_edges("run_code", after_run, ["report", "fix_code", "give_up"])
    graph.add_edge("fix_code", "run_code")
    graph.add_edge("report", END)
    graph.add_edge("give_up", END)
    return graph.compile(name="analytics_agent")
