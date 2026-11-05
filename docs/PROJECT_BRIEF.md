# Project: OrchestraOS — a free, local multi-agent AI operating system

> Original project brief, saved verbatim. NOTE: the brief was truncated at
> "Repo structure (target)" — the repo structure and the phase list are still
> to be added.

## Context (carry forward)
- CS student portfolio project for AI/ML roles. Must be explainable in an interview: prefer clear, well-structured code over clever code.
- HARD CONSTRAINT: everything must be 100% free. No paid APIs, no paid hosting, no credit card. LLMs run locally via Ollama. Never add a dependency or service that requires payment.
- OS: a normal laptop (Windows/Mac/Linux) with 8–16 GB RAM. Everything must run with `docker compose up` + a few commands.

## What the system does
A user gives a goal in a chat UI (e.g. "Research LangGraph vs CrewAI, analyse this CSV, and write me a report with a chart"). A supervisor agent plans the task, routes steps to specialist agents, keeps memory across sessions, uses tools through MCP, asks for human approval before side-effecting actions, and returns a final answer with sources.

## Architecture
- **UI:** Streamlit chat app → talks to FastAPI backend (SSE streaming).
- **Orchestrator:** LangGraph `StateGraph` supervisor. Typed state (goal, plan, steps, results, messages, approvals). SQLite checkpointer so runs resume after restart.
- **Specialist agents** (each its own LangGraph subgraph with its own prompt):
  1. Memory agent — retrieves context before planning and stores new facts/summaries after.
  2. Research agent — web search (DuckDuckGo, no API key) + reading local documents; returns findings with source URLs.
  3. Writing agent — drafts reports/answers in markdown from research + analysis results.
  4. Analytics agent — writes and runs pandas/matplotlib code inside a Docker sandbox (no network, resource limits); returns numbers + chart files.
  5. Action agent — file writes and other side effects, ALWAYS gated by LangGraph `interrupt` for human approval in the UI.
- **Memory layer:** ChromaDB (persistent local) for semantic memory using `nomic-embed-text` via Ollama; Neo4j Community (Docker) for entities/relationships (user → prefers → X, project → uses → Y).
- **Tool layer:** tools exposed as MCP servers using the official `mcp` Python SDK (FastMCP), loaded into agents via `langchain-mcp-adapters`. At minimum: search server, filesystem server (sandboxed to a `workspace/` folder), python-sandbox server.
- **Model router:** picks the model per step — small model (~3B) for routing/classification, larger (~7–8B tool-calling instruct model) for planning/writing. Model names come from `.env`, never hardcoded. Fallback: if a model call fails or returns invalid JSON, retry once, then fall back to the other model.
- **A2A:** the Research agent is ALSO exposed as a standalone A2A service using the official `a2a-sdk` (Agent Card at `/.well-known/agent-card.json`); the supervisor can call it over A2A when `RESEARCH_MODE=a2a`.
- **Observability:** Langfuse (self-hosted in Docker) traces every agent step.
- **Evals:** a fixed benchmark of 30+ tasks with pass/fail checks, so all metrics in the README are measured, not invented.

## Repo structure (target)
_TODO — missing from the brief._

## Phases
_TODO — missing from the brief._
