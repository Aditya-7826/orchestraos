# OrchestraOS. Recipes avoid shell-specific syntax so they work from cmd, PowerShell and bash.

.PHONY: help setup sandbox-image doctor dev ui a2a-research test lint up up-graph up-observability down logs ps

help:
	@echo "make setup     - install deps, create .env and workspace/ (safe to re-run)"
	@echo "make sandbox-image - build the no-network image analytics code runs in"
	@echo "make doctor    - check Ollama, models, Docker, sandbox image and RAM"
	@echo "make dev       - run the API locally with reload on http://localhost:8000"
	@echo "make ui        - run the Streamlit UI locally on http://localhost:8501"
	@echo "make a2a-research - run the research agent as an A2A service on :8100"
	@echo "make test      - run the test suite"
	@echo "make lint      - ruff lint + format check"
	@echo "make up        - start api + ui in Docker (UI on http://localhost:8501)"
	@echo "make up-graph  - core stack + Neo4j"
	@echo "make up-observability - Langfuse tracing UI on http://localhost:3000"
	@echo "make down      - stop everything"

setup:
	uv sync
	uv run python scripts/setup_env.py

sandbox-image:
	docker build -t orchestraos-sandbox:latest sandbox

doctor:
	uv run orchestraos-doctor

dev:
	uv run uvicorn orchestraos.api.main:app --reload

ui:
	uv run streamlit run ui/app.py

a2a-research:
	uv run uvicorn orchestraos.a2a.research_service:app --port 8100

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

up:
	docker compose up -d --build

up-observability:
	docker compose --profile observability up -d langfuse-web langfuse-worker
	@echo "Langfuse: http://localhost:3000 (login: LANGFUSE_UI_EMAIL / LANGFUSE_UI_PASSWORD in .env); set TRACING=true"

up-graph:
	docker compose --profile graph up -d --build

down:
	docker compose --profile graph --profile a2a --profile observability down

logs:
	docker compose logs -f

ps:
	docker compose ps
