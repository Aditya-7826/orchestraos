# OrchestraOS. Recipes avoid shell-specific syntax so they work from cmd, PowerShell and bash.

.PHONY: help setup sandbox-image doctor dev test lint up up-graph down logs ps

help:
	@echo "make setup     - install deps (uv) and create .env from .env.example"
	@echo "make sandbox-image - build the no-network image analytics code runs in"
	@echo "make doctor    - check Ollama, models, Docker, sandbox image and RAM"
	@echo "make dev       - run the API locally with reload on http://localhost:8000"
	@echo "make test      - run the test suite"
	@echo "make lint      - ruff lint + format check"
	@echo "make up        - start the core stack (docker compose)"
	@echo "make up-graph  - core stack + Neo4j"
	@echo "make down      - stop everything"

setup:
	uv sync
	uv run python -c "import shutil, pathlib; p = pathlib.Path('.env'); p.exists() or shutil.copy('.env.example', p)"

sandbox-image:
	docker build -t orchestraos-sandbox:latest sandbox

doctor:
	uv run orchestraos-doctor

dev:
	uv run uvicorn orchestraos.api.main:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

up:
	docker compose up -d --build

up-graph:
	docker compose --profile graph up -d --build

down:
	docker compose --profile graph down

logs:
	docker compose logs -f

ps:
	docker compose ps
