# OrchestraOS

A free, local multi-agent AI operating system. You give a goal in a chat UI; a LangGraph
supervisor plans it, routes steps to specialist agents (memory, research, writing, analytics,
action), uses tools over MCP, asks for your approval before side effects, and answers with sources.

Everything runs locally: LLMs via [Ollama](https://ollama.com), no paid APIs, no cloud services.

> **Status: Phase 6 of 10.** Chat UI -> streaming API -> supervisor -> research, analytics,
> writing and action agents work end to end on local models, with human approval before any
> file is written. Memory, A2A, tracing and evals come next. See [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) for the phase plan.

## How a run works

```
goal -> recall_memory -> triage (small model: "direct" or "plan")
     -> plan (large model: JSON list of steps, last one is always "writing")
     -> research / analytics / action / writing nodes, one per step, in plan order
     -> finalize (answer + de-duplicated sources) -> store_memory
```

State is checkpointed to SQLite after every node, so a run that stops part-way resumes
where it left off: `uv run orchestraos run --thread <id>`.

## Quick start

Prerequisites (all free): Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, Ollama.

```bash
make setup                      # uv sync, create .env and workspace/ (with sample data)
ollama pull qwen2.5:3b          # small model: routing
ollama pull qwen2.5:7b          # large model: planning / writing
ollama pull nomic-embed-text    # embeddings
make sandbox-image              # build the Python sandbox image (one-off)
make doctor                     # checks Ollama, models, Docker, sandbox image, RAM
make up                         # api + ui in Docker
# open http://localhost:8501 and try: "Analyse sales.csv: revenue per region with a chart"
```

No `make`? Every target is a single command; see the [Makefile](Makefile).

## Running on 8 GB of RAM

Ollama runs **on the host**, not in Docker, so the model's memory isn't counted against the
Docker VM. Heavier services are behind Compose profiles so you only run what you need:

| Command | Services |
|---|---|
| `docker compose up -d` | `api` (FastAPI, MCP servers) + `ui` (Streamlit) |
| `docker compose --profile graph up -d` | + Neo4j (capped at 1 GB) |

`make doctor` warns when the machine has less than 12 GB of RAM. On those machines, enable
optional profiles one at a time.

## Layout

```
src/orchestraos/
  config.py      # settings from .env (model names are required, never hardcoded)
  ollama.py      # Ollama REST helpers: list pulled models, find missing ones
  doctor.py      # `orchestraos-doctor` setup checks
  cli.py         # `orchestraos run "<goal>"`
  llm/router.py  # small vs large model per step, retry once, then fall back
  graph/         # supervisor StateGraph, typed state, planner, checkpointer, answer clean-up
  agents/        # research, analytics, writing (+ action stub); each a LangGraph subgraph
  memory/        # memory interface
  tools/         # tool logic: workspace files, web search/fetch, Docker sandbox, MCP client
  mcp_servers/   # FastMCP servers (stdio): search, filesystem, python_sandbox
  api/           # FastAPI: runs as background tasks, SSE events, workspace files
ui/app.py        # Streamlit chat UI (talks to the API only)
scripts/         # setup helper
sandbox/         # Dockerfile for the no-network Python sandbox
examples/        # sample data (copy into workspace/ to try the analytics agent)
tests/           # pytest; LLMs are faked, Docker tests skip if the sandbox image is missing
docs/            # project brief
```

## Development

```bash
make test    # pytest
make lint    # ruff check + format check
make dev     # API with auto-reload on :8000 (host; uses your Docker for the sandbox)
make ui      # Streamlit UI on :8501, talking to the API on :8000
```

`GET /health` returns `ok` when Ollama is reachable and every configured model is pulled,
otherwise `degraded` with the reason (the API itself is still up, so it returns 200).

## API

| Endpoint | |
|---|---|
| `POST /runs {goal}` | start a run in the background -> `{thread_id}` |
| `GET /runs/{id}/events?after=n` | SSE: `triage`, `plan`, `step`, `token` (writer, live), `interrupt` (approval needed), `final`, `error`, `done`; replays from event `n` |
| `GET /runs/{id}` | state from the checkpointer (works after an API restart) |
| `POST /runs/{id}/resume` | continue a run that stopped part-way |
| `POST /runs/{id}/approval {approved, note}` | answer a pending approval; the run continues |
| `GET/POST /files`, `GET /files/{path}` | list, upload, download workspace files (path-checked) |
| `GET /health` | Ollama reachable + models pulled |

Runs don't live inside the HTTP request: they take minutes on a laptop, and Streamlit re-runs
its script on every interaction. The UI follows a run's event stream and reconnects from the
last event it saw.

In Docker, the API starts sandbox containers through the host's Docker socket (`make setup`
writes the workspace's host path to `.env` as `HOST_WORKSPACE`). That gives the API container
control of Docker: fine for a local single-user app, not for anything shared.

## Tools and sandboxing

Tools live in three MCP servers (official `mcp` SDK, stdio). The app starts each one once and
loads its tools through `langchain-mcp-adapters`.

| Server | Tools | Guard rails |
|---|---|---|
| `search` | `web_search`, `fetch_page` | public http(s) only; every redirect hop re-checked (no SSRF to localhost / private IPs) |
| `filesystem` | `list_files`, `read_file` (text, PDF), `write_file` | paths resolved and confined to `workspace/` |
| `python_sandbox` | `run_python` | Docker: `--network none`, 512 MB, 1 CPU, 128 pids, read-only root, no capabilities, workspace read-only at `/data`, output to `workspace/outputs/<run>/` |

## Agents

Each agent is a fixed LangGraph pipeline, not a free-form tool-calling loop. With 7B local
models, tool-calling loops are unreliable; giving each LLM call one narrow job (write search
queries, write code, write prose) and letting code make the tool calls works much better.

| Agent | Pipeline | Output |
|---|---|---|
| research | plan queries -> search + fetch pages (and local docs named in the task) -> synthesize | findings with `[n]` citations + source URLs |
| analytics | find data -> profile it in the sandbox -> write code -> run -> fix on error (max 2) -> report | printed results + chart files |
| writing | draft | markdown answer |
| action | propose writes -> **`interrupt()`: wait for the user** -> write only if approved | saved file paths, or "not saved" |

After the writer, code (not the model) has the last word on the answer:
- image links to files no step produced are removed, and charts the writer forgot are appended
- if any step failed, a notice saying so goes at the top, so made-up figures can't pass as real

### Human approval for side effects

The action agent is the only agent that can call `write_file` (each agent gets a restricted view
of the tools), and it calls LangGraph's `interrupt()` before writing. The run pauses and is
checkpointed; the UI shows every proposed write (path, size, overwrite warning, preview) with
Approve / Reject; `POST /runs/{id}/approval` resumes the run with the decision. The decision is
recorded in the run's state (`approvals`) and reported in the answer.

To save a long answer, the model doesn't retype it: an action says "save step N's output" and
code copies it exactly. (When the model retyped it anyway, it saved a 200-character fragment, so
content that matches an earlier step is now replaced with that step's full output.) If the goal
asks to save to a file but the planner leaves out the action step, code adds it.

### What the first live runs taught us

These came from running real goals on `qwen2.5:7b` / `qwen2.5:3b` on an 8 GB laptop:

- **The model guessed column types** from raw CSV lines and wrote code that summed a date
  column. Fix: a fixed profiling script runs first, so the model sees real dtypes.
- **pandas 3 broke generated code** (the `"M"` month alias was removed); the model couldn't fix
  it even with the error message. Fix: the sandbox pins pandas 2.x.
- **The writer invented results for a failed step**, despite being told not to. Fix: failed
  steps are described plainly, and the failure notice above is added by code.
- **Speed:** the 7B model generated ~2–5 tokens/s (CPU/GPU split); a smaller context window
  (6144 instead of 8192) was noticeably faster. A CSV analysis + report took ~2 min; a goal with
  web research + analysis + report took ~14 min. Phase 10 measures this properly.
