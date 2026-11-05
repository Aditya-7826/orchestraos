"""Small helpers for the local Ollama server's REST API."""

from collections.abc import Iterable

import httpx


class OllamaUnavailable(Exception):
    """Raised when the Ollama server can't be reached or returns an error."""


def normalize_model_name(name: str) -> str:
    """Ollama treats `name` and `name:latest` as the same model."""
    return name if ":" in name else f"{name}:latest"


def list_local_models(
    base_url: str, *, client: httpx.Client | None = None, timeout: float = 3.0
) -> set[str]:
    """Return the (normalized) names of every model pulled into the local Ollama."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        if client is None:
            with httpx.Client(timeout=timeout) as own_client:
                response = own_client.get(url)
        else:
            response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaUnavailable(f"cannot reach Ollama at {base_url}: {exc}") from exc

    return {normalize_model_name(m["name"]) for m in response.json().get("models", [])}


def missing_models(required: Iterable[str], available: set[str]) -> list[str]:
    """Models from `required` that aren't in `available`, keeping the caller's spelling."""
    return [name for name in required if normalize_model_name(name) not in available]
