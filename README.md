# OrchestraOS

A free, local multi-agent AI operating system. You give a goal in a chat UI; a LangGraph
supervisor plans it, routes steps to specialist agents (memory, research, writing, analytics,
action), uses tools over MCP, asks for your approval before side effects, and answers with sources.

Everything runs locally: LLMs via [Ollama](https://ollama.com), no paid APIs, no cloud services.

> **Status: Phase 3 of 10.** Model router, supervisor graph and MCP tool servers (search,
> workspace files, Docker Python sandbox) work against local models. Real agents, memory, UI,
> A2A, tracing and evals come next. See [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) for the design and phase plan.

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
make setup                      # uv sync + create .env from .env.example
ollama pull qwen2.5:3b          # small model: routing
ollama pull qwen2.5:7b          # large model: planning / writing
ollama pull nomic-embed-text    # embeddings
make sandbox-image              # build the Python sandbox image (one-off)
make doctor                     # checks Ollama, models, Docker, sandbox image, RAM
make up                         # core stack
curl http://localhost:8000/health
```

No `make`? Every target is a single command; see the [Makefile](Makefile).

## Running on 8 GB of RAM

Ollama runs **on the host**, not in Docker, so the model's memory isn't counted against the
Docker VM. Heavier services are behind Compose profiles so you only run what you need:

| Command | Services |
|---|---|
| `docker compose up -d` | `api` |
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
  graph/         # supervisor StateGraph, typed state, planner, SQLite checkpointer
  agents/        # agent contract (each agent is a LangGraph subgraph)
  memory/        # memory interface
  tools/         # tool logic: workspace files, web search/fetch, Docker sandbox, MCP client
  mcp_servers/   # FastMCP servers (stdio): search, filesystem, python_sandbox
sandbox/         # Dockerfile for the no-network Python sandbox
  api/main.py    # FastAPI app: GET /health
tests/           # pytest, no network or Docker needed
docs/            # project brief
```

## Development

```bash
make test    # pytest
make lint    # ruff check + format check
make dev     # API with auto-reload on :8000
```

`GET /health` returns `ok` when Ollama is reachable and every configured model is pulled,
otherwise `degraded` with the reason (the API itself is still up, so it returns 200).

## Tools and sandboxing

Tools live in three MCP servers (official `mcp` SDK, stdio). The app starts each one once and
loads its tools through `langchain-mcp-adapters`.

| Server | Tools | Guard rails |
|---|---|---|
| `search` | `web_search`, `fetch_page` | public http(s) only; every redirect hop re-checked (no SSRF to localhost / private IPs) |
| `filesystem` | `list_files`, `read_file` (text, PDF), `write_file` | paths resolved and confined to `workspace/` |
| `python_sandbox` | `run_python` | Docker: `--network none`, 512 MB, 1 CPU, 128 pids, read-only root, no capabilities, workspace read-only at `/data`, output to `workspace/outputs/<run>/` |
