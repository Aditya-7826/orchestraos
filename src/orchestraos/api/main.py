"""FastAPI backend. Phase 0 only exposes a health check; chat/SSE endpoints come later."""

from typing import Annotated, Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from orchestraos import __version__
from orchestraos.config import Settings, get_settings
from orchestraos.ollama import OllamaUnavailable, list_local_models, missing_models

app = FastAPI(title="OrchestraOS API", version=__version__)


class OllamaStatus(BaseModel):
    reachable: bool
    missing_models: list[str] = []
    error: str | None = None


class HealthResponse(BaseModel):
    # "degraded" = the API is up but can't do useful work (Ollama down or models missing).
    status: Literal["ok", "degraded"]
    version: str
    ollama: OllamaStatus


@app.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    # A plain `def` endpoint runs in FastAPI's threadpool, so the blocking HTTP call is fine.
    try:
        available = list_local_models(settings.ollama_base_url)
    except OllamaUnavailable as exc:
        return HealthResponse(
            status="degraded",
            version=__version__,
            ollama=OllamaStatus(reachable=False, error=str(exc)),
        )

    missing = missing_models(settings.required_models(), available)
    return HealthResponse(
        status="degraded" if missing else "ok",
        version=__version__,
        ollama=OllamaStatus(reachable=True, missing_models=missing),
    )
