# OrchestraOS

A free, local multi-agent AI operating system. You give a goal in a chat UI; a LangGraph
supervisor plans it, routes steps to specialist agents (memory, research, writing, analytics,
action), uses tools over MCP, asks for your approval before side effects, and answers with sources.

Everything runs locally: LLMs via [Ollama](https://ollama.com), no paid APIs, no cloud services.

> **Status: Phase 0 (foundation).** Config, health check, setup doctor, Docker Compose with
> profiles. Agents, memory, MCP tools, UI, A2A, tracing and evals come in later phases.
> See [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) for the full design.

## Quick start

Prerequisites (all free): Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, Ollama.

```bash
make setup                      # uv sync + create .env from .env.example
ollama pull qwen2.5:3b          # small model: routing
ollama pull qwen2.5:7b          # large model: planning / writing
ollama pull nomic-embed-text    # embeddings
make doctor                     # checks Ollama, models, Docker, RAM
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
