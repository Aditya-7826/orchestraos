FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /usr/local/bin/uv
# Docker CLI only (no daemon): the python_sandbox MCP server starts sandbox containers
# through the host's Docker daemon via the mounted socket.
COPY --from=docker:29-cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY ui ./ui
RUN uv sync --frozen --no-dev

EXPOSE 8000 8501
CMD ["uvicorn", "orchestraos.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
