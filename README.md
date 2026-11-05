# OrchestraOS

A free, local multi-agent AI system. You give it a goal in a chat UI, such as *"Research
LangGraph vs CrewAI, analyse sales.csv and write me a report with a chart"*. A LangGraph
supervisor plans the work and routes each step to a specialist agent (research, analytics,
writing, action). The agents use tools over MCP. You approve any file write before it
happens, the system remembers you across sessions, and the answer comes back with its sources.

Everything runs on your machine: models via [Ollama](https://ollama.com) (`qwen2.5:7b`,
`qwen2.5:3b`, `nomic-embed-text`), with no paid APIs or cloud services and no API keys.

**Measured result:** 35 of 36 benchmark tasks passed (97%) on an 8 GB Windows laptop, with
the full suite taking 51 minutes. See [Results](#results).

## What it does

- **Plans and delegates.** A supervisor graph triages each goal with a small model, plans it
  with a large one, and runs one specialist agent per step. State is checkpointed, so a run
  that stops part-way resumes where it left off.
- **Researches with sources.** It searches the web (DuckDuckGo, no API key) and reads the
  pages and any local documents named in the goal, then cites them as numbered sources.
- **Analyses data safely.** It writes pandas/matplotlib code and runs it in a Docker sandbox
  with no network, capped resources and the data mounted read-only. Errors go back to the model
  to fix (up to 2 times).
- **Asks before acting.** The only agent that can write files pauses the run with LangGraph's
  `interrupt()` and waits for Approve or Reject in the UI.
- **Remembers you.** Semantic memory (ChromaDB) and graph memory (Neo4j, optional) keep your
  stated preferences and past work across sessions.
- **Speaks open protocols.** Its tools are MCP servers (official `mcp` SDK), and the research
  agent is also served over A2A (official `a2a-sdk`) with an Agent Card.
- **Is observable and measured.** Every step can be traced in self-hosted Langfuse, and a
  36-task benchmark with pass/fail checks produces the numbers below.

## Architecture

```
 Streamlit chat UI ──HTTP/SSE──▶ FastAPI (runs as background tasks, event replay)
                                   │
                                   ▼
                    LangGraph supervisor  (SQLite checkpointer)
   recall_memory ─▶ triage (3B) ─▶ plan (7B) ─▶ [ one agent per step ] ─▶ finalize ─▶ store_memory
                                                 │
        ┌──────────────────┬─────────────────────┼─────────────────────┐
     research           analytics             writing                action
   (local or A2A)    (code + sandbox)       (markdown)       (interrupt → approve)
        │                  │                                          │
        ▼                  ▼                                          ▼
   MCP: search,       MCP: python_sandbox ──▶ Docker               MCP: filesystem
   filesystem         (no network, read-only data)                 (workspace/ only)

   Memory: ChromaDB (+ Neo4j)   ·   Models: Ollama on the host   ·   Tracing: Langfuse
```

## Results

The benchmark has 36 tasks with machine-checked pass/fail criteria
([evals/tasks.yaml](evals/tasks.yaml)). Every number here comes from running it, and the raw
per-task results are in [evals/results/](evals/results/).

| Category | What it checks | Passed | Median time |
|---|---|---|---|
| triage | the small model picks "direct answer" vs "plan" correctly | 8/8 | <1 s |
| planning | the plan uses the right agents (e.g. no web research for a CSV question) | 8/8 | 15 s |
| analytics | exact numbers computed from the fixture CSVs, charts produced | 6/7 | 136 s |
| grounding | no invented results for a missing file, no made-up sources | 3/3 | 72 s |
| safety | writes only after approval; rejected and out-of-workspace writes don't happen; overwrites are flagged | 5/5 | 82 s |
| research | real web research returns real sources; local documents are read and cited | 3/3 | 212 s |
| memory | a preference stated in one session shapes a later, unrelated one | 2/2 | 178 s |
| **all** | | **35/36 (97%)** | 60 s |

The models were `qwen2.5:7b` and `qwen2.5:3b` with `nomic-embed-text`, on Windows 11 with
7.3 GB RAM. The full run took 51 minutes (commit `0ee994e`). On top of its own checks, every
end-to-end task also checks that the answer has no URL or image that no tool produced.

**How the numbers got here.** The benchmark was also used during development, so two fixes came
from it:
- **First smoke run:** the planner sent a markdown notes file to the CSV agent. The fix was a
  prompt rule saying which agent handles which file types.
- **First full run, 34/36:** a stated preference wasn't recalled for an unrelated question,
  because similarity search alone can't connect the two. Facts about the user are now always
  recalled. That run also turned up a CSV rendered as a broken image.

The second full run, on the fixed code, gave the 35/36 above. Both runs' results are committed.

**The remaining failure** (`a-early-starters`) fails the same way in both runs. Asked for the
employees who started *before 2021*, the 7B model leaves out both 2020 starters, an off-by-one
in the filter it writes. It's left visible rather than tuned away.

Reproduce with `uv run python -m evals.run` (about an hour on a laptop). The runner is
resumable (`--resume`), runs each task in a fresh workspace and memory store, and warns if
you're on battery or low on RAM.

## Quick start

Prerequisites (all free): Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, Ollama.

```bash
make setup                      # deps, .env (with generated secrets), workspace/ with sample data
ollama pull qwen2.5:3b          # small model: triage, fact extraction
ollama pull qwen2.5:7b          # large model: planning, code, writing
ollama pull nomic-embed-text    # embeddings for memory
make sandbox-image              # the no-network Python sandbox image (one-off)
make doctor                     # checks Ollama, models, Docker, sandbox image, RAM
make demo                       # guided step-by-step tour (see the walkthrough below)
make up                         # api + ui in Docker
# open http://localhost:8501 and try: "Analyse sales.csv: revenue per region with a chart"
```

Or without Docker for the app itself: `make dev` (API on :8000) and `make ui` (UI on :8501).
A terminal run is `uv run orchestraos run "<goal>"`. If you don't have `make`, each target is
a single command; see the [Makefile](Makefile).

## Step-by-step walkthrough

### 1. Take the guided tour (about 10 minutes)

```bash
make demo          # or: uv run orchestraos demo   (add --yes to skip the pauses)
```

The tour runs five real goals through the whole system. Before each one it explains what is
about to happen, and afterwards it tells you what to look for. It keeps its own memory in
`data/demo/`, so it doesn't change yours. Here is what each step does, with excerpts from a real
run on an 8 GB laptop:

**Step 1: check the setup.** These are the same checks as `make doctor`. If something is
missing, the tour stops and prints the fix (for example `ollama pull qwen2.5:7b`).

**Step 2: ask a simple question.** The small model decides no tools are needed, so the large
model answers in a single writing step:

```
> "In two sentences, what does a multi-agent AI system do?"
  triage (small model): direct answer
  plan (large model):
    1. writing: In two sentences, what does a multi-agent AI system do?
```

**Step 3: analyse a data file.** The analytics agent profiles `sales.csv`, writes
pandas/matplotlib code and runs it in the no-network Docker sandbox:

```
> "Analyse sales.csv: total revenue per region, with a bar chart"
  triage (small model): needs a plan
  plan (large model):
    1. analytics: Load 'sales.csv', calculate total revenue per region, generate a bar chart.
    2. writing: Present the total revenue per region and the bar chart in markdown.
  step 1 analytics: ok - ... region North 106080 South 79920 ...
  Answer:
    | Region | Total Revenue |
    | North  | 106,080       |
    | South  | 79,920        |
  File: ...\workspace\outputs\6df147bd40\revenue_by_region.png
```

**Step 4: approve a side effect.** Saving a file pauses the run until you answer:

```
> "Write a 3-item checklist for reviewing a pull request and save it to notes/demo-checklist.md"
  plan (large model):
    1. writing: Write a 3-item checklist for reviewing a pull request ...
    2. action: Save the answer from the previous step to notes/demo-checklist.md
  The run is paused. It wants to make these changes:
    write_file -> notes/demo-checklist.md, 579 chars
      | # Checklist for Reviewing Pull Requests
      | ...
  Approve? [y/N] y
  step 2 action: ok - Saved to notes/demo-checklist.md (approved by you).
```

Answer `n` and nothing is written: the answer says "Not saved".

**Step 5: memory across sessions.** You state a preference, then ask something unrelated in a
new session:

```
> "I prefer answers as short bullet points. What is Docker?"
> "What is Kubernetes?"
  Recalled from memory:
    Known about the user:
    - user prefers answers as short bullet points
  Answer:
    - Kubernetes is an open-source container orchestration tool.
    - It automates the deployment, scaling, and management of containerized applications.
    ...
```

### 2. Use the chat UI

1. Start the app: `make up` (Docker), or `make dev` and `make ui` in two terminals.
2. Open http://localhost:8501. The sidebar shows whether the models are ready.
3. Add a file: drag a CSV, PDF or text file onto **Add a file** in the sidebar. It goes into
   `workspace/`. `sales.csv` and `employees.csv` are there already.
4. Type a goal in the chat box, for example *"Analyse employees.csv: average salary per
   department, with a chart"*.
5. Watch it work. The status box shows the plan and each finished step, and the answer streams
   in as the writer produces it. Charts appear inline.
6. Approve or reject. If the goal asks to save something (*"... and save it to
   reports/salaries.md"*), the run stops and shows each proposed file with a preview. Click
   **Approve** or **Reject**, adding a note if you like.
7. Check the sources. Expand **Sources** under the answer to see the URLs and files used, and
   **How I did it** to see the plan and each step's result.
8. Come back later. Tell it a preference once (*"I prefer short bullet points"*) and later
   answers follow it.

### 3. Use it from the terminal or the API

```bash
uv run orchestraos run "Research what LangGraph checkpointers are used for"
uv run orchestraos run --thread <id>         # resume a run that stopped part-way

curl -X POST localhost:8000/runs -H 'content-type: application/json' \
     -d '{"goal": "Analyse sales.csv: total units per region"}'      # -> {"thread_id": "..."}
curl -N localhost:8000/runs/<thread_id>/events                       # live progress (SSE)
curl -X POST localhost:8000/runs/<thread_id>/approval \
     -H 'content-type: application/json' -d '{"approved": true}'     # answer an approval
```

### Running on 8 GB of RAM

Ollama runs on the host, not in Docker, so the model's memory isn't counted against the Docker
VM. Heavier services sit behind Compose profiles so you only run what you need:

| Command | Adds |
|---|---|
| `make up` | `api` (FastAPI + MCP servers) and `ui` (Streamlit) |
| `make up-graph` | Neo4j graph memory (capped at 1 GB) |
| `docker compose --profile a2a up -d` | the research agent as an A2A service on :8100 |
| `make up-observability` | Langfuse tracing (6 containers, ~1.6 GB idle) |

## How it works

### A run

```
goal -> recall_memory -> triage (small model: "direct" or "plan")
     -> plan (large model: JSON steps, ending with "writing", or "writing" then "action")
     -> one agent node per step, in plan order
     -> finalize (answer clean-up, de-duplicated sources) -> store_memory
```

The **model router** sends routing, classification and extraction to the 3B model, and
planning, code and writing to the 7B model. If a call fails or returns invalid JSON, the router
retries once, then falls back to the other model. Model names come from `.env`, never from code.

### Agents

Each agent is a fixed LangGraph subgraph, not a free-form tool-calling loop. 7B local models
are unreliable at long tool-calling loops. Giving each LLM call one narrow job (write search
queries, write code, write prose) and letting code make the tool calls works much better.

| Agent | Pipeline | Output |
|---|---|---|
| research | plan queries → search + fetch pages (+ local docs named in the goal) → synthesize | findings with `[n]` citations and sources |
| analytics | find data → profile it in the sandbox → write code → run → fix on error (max 2) → report | printed results + chart files |
| writing | draft | the markdown answer |
| action | propose writes → **`interrupt()`: wait for the user** → write only if approved | saved paths, or "not saved" |

Each agent gets a restricted view of the tools. Only the action agent can call `write_file`.

### Code checks the answer, not the model

After the writer, `finalize` applies these rules in code:
- **Made-up links:** image links to files no step produced are removed, and real charts the
  writer forgot are appended.
- **Made-up sources:** URLs no tool returned are removed, along with citation numbers when
  nothing was researched.
- **Failed steps:** if any step failed, a notice at the top says so, so invented figures can't
  pass as real ones.

Each of these was added after a live run showed the 7B model doing the opposite.

### Human approval

The action agent proposes writes, then calls `interrupt()`. The run pauses and is checkpointed.
The UI shows every proposed write (path, size, overwrite warning, preview), and
`POST /runs/{id}/approval` resumes the run with your decision, which is recorded in the run's
state. To save a long answer, the model doesn't retype it: the action says "save step N's
output" and code copies it exactly.

### Memory

- **Semantic (ChromaDB, `nomic-embed-text`):** past goals and answers, plus facts you stated.
  The similarity cut-offs are calibrated on real embeddings: related past runs scored 0.13–0.16
  cosine distance and unrelated ones 0.49–0.58.
- **Facts about you are always recalled:** your preferences apply to every goal, however
  unrelated to the topic.
- **Graph (Neo4j, optional):** facts are stored as `user -[prefers]-> bullet points`.
- **Facts come only from your own words:** the small model extracts them from your goal, never
  from an answer, so a hallucination can't become a memory.
- **Best-effort:** if a memory store is down, the run carries on without it.

### Tools and sandboxing (MCP)

There are three MCP servers (FastMCP over stdio). The app starts each one once and loads its
tools through `langchain-mcp-adapters`.

| Server | Tools | Guard rails |
|---|---|---|
| `search` | `web_search`, `fetch_page` | public http(s) only; every redirect hop re-checked (no SSRF to localhost or private IPs) |
| `filesystem` | `list_files`, `read_file` (text, PDF), `write_file` | paths resolved and confined to `workspace/` |
| `python_sandbox` | `run_python` | Docker: `--network none`, 512 MB, 1 CPU, 128 pids, read-only root, no capabilities, data read-only at `/data` |

In Docker, the API starts sandbox containers through the host's Docker socket. That gives the
API container control of Docker, which is fine for a local single-user app but not for anything
shared.

### Research over A2A

```bash
make a2a-research                                       # serves on :8100
curl http://localhost:8100/.well-known/agent-card.json
RESEARCH_MODE=a2a uv run orchestraos run "Research what the A2A protocol is"
```

With `RESEARCH_MODE=a2a`, the supervisor's research step sends an A2A message and reads the
`findings` artifact from the completed task. It's the same research subgraph with a different
transport, and the same `AgentState` goes in and out.

### Tracing (Langfuse)

Self-hosted Langfuse 4.75 runs in the `observability` profile. It's the official stack minus
the optional AI gateway, with telemetry off, only the UI published (localhost:3000), and memory
caps. `make setup` generates every secret and the API keys, so there's no sign-up step. Run
`make up-observability`, then set `TRACING=true`.

Each run is one trace: every supervisor node, agent subgraph, MCP tool call and LLM call (with
model, tokens and latency). A thread's traces share a Langfuse session, so a run paused for
approval stays together.

### API

| Endpoint | |
|---|---|
| `POST /runs {goal}` | start a run in the background → `{thread_id}` |
| `GET /runs/{id}/events?after=n` | SSE events: `triage`, `plan`, `step`, `token`, `interrupt`, `final`, `error`, `done`; replays from event `n` |
| `GET /runs/{id}` | run state from the checkpointer (works after an API restart) |
| `POST /runs/{id}/resume` | continue a run that stopped part-way |
| `POST /runs/{id}/approval {approved, note}` | answer a pending approval |
| `GET/POST /files`, `GET /files/{path}` | list, upload, download workspace files |
| `GET /health` | Ollama reachable and models pulled |

A run doesn't live inside the HTTP request: it takes minutes on a laptop, and Streamlit
re-runs its script on every interaction. The UI follows a run's event stream and reconnects
from the last event it saw.

## What building it taught me

These are real failures from live runs on an 8 GB laptop, and what fixed them:

- **The model guessed column types** from raw CSV lines and summed a date column. Now a fixed
  profiling script runs first, so the model sees the real types.
- **pandas 3 broke generated code**, because the model writes pandas 2 code (for example the
  `"M"` alias). The sandbox now pins pandas 2.x.
- **The writer invented results for a failed step** and made up a source URL. Now code adds a
  failure notice and drops unbacked URLs.
- **The model retyped a long answer it was asked to save** and saved a 200-character fragment.
  Now saves copy the earlier step's output exactly.
- **LangGraph's `GraphInterrupt` is an `Exception` subclass**, so the "record agent errors"
  handler swallowed approval pauses. The handler now re-raises `GraphBubbleUp`.
- **The end of each trace went missing:** the 5 s export timeout aborted the last batch while
  the model saturated the CPU. The export timeout is now 30 s.
- **Speed:** the 7B model generated about 2–5 tokens/s, split between CPU and GPU. A 6144-token
  context was noticeably faster than 8192.

## Limitations

- **Small models make reasoning mistakes:** the benchmark failure above is one, and the code
  checks catch made-up links, sources and figures but not wrong logic in correct-looking code.
  A wrong fact in an answer with no sources (e.g. confusing vector databases with GIS vector
  data) also gets through.
- **It's slow on a laptop:** a CSV question takes about 2 minutes and a research question 3–5
  minutes. Generation is the bottleneck at 2–5 tokens/s.
- **The benchmark is small and was run twice:** 36 tasks give a clear signal, but LLM output
  varies, and repeated runs would show the spread.
- **It's single-user and local:** in Docker the API controls the host's Docker daemon to start
  sandboxes, and there's no authentication. Don't expose it to a network.
- **Web research depends on DuckDuckGo:** results change over time, and it can rate-limit
  heavy use.

## Development

```bash
make test    # pytest: LLMs are faked; Docker tests skip if the sandbox image is missing
make lint    # ruff check + format check
uv run pytest -m live          # also runs the live tests against Ollama
uv run python -m evals.run     # the benchmark (see Results)
```

```
src/orchestraos/
  config.py        settings from .env
  demo.py          `orchestraos demo`: the guided step-by-step tour
  llm/router.py    small vs large model per step, retry, fall back
  graph/           supervisor graph, state, planner, checkpointer, answer checks
  agents/          research (local or A2A), analytics, writing, action
  memory/          memory agent: ChromaDB + Neo4j, recall/remember subgraphs
  tools/           workspace files, web search/fetch, Docker sandbox, MCP client
  mcp_servers/     FastMCP servers: search, filesystem, python_sandbox
  a2a/             research agent as an A2A service
  api/             FastAPI: background runs, SSE, files
  observability.py Langfuse tracing (opt-in)
ui/app.py          Streamlit chat UI
evals/             benchmark tasks, checks, runner, results
sandbox/           Dockerfile for the Python sandbox
examples/          sample data and notes used by the demo and the benchmark
docs/              the original project brief and phase plan
```
