"""Application settings, loaded from environment variables and `.env`.

Model names are deliberately required with no defaults: the brief says they
must come from `.env`, never from code, so a missing one fails loudly at
startup instead of silently using a model you haven't pulled.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLMs (served by Ollama on the host) ---
    ollama_base_url: str = "http://localhost:11434"
    small_model: str  # ~3B: routing / classification
    large_model: str  # ~7-8B tool-calling instruct model: planning / writing
    embed_model: str  # embeddings for semantic memory

    # --- Agents ---
    # "local": supervisor calls the research subgraph in-process.
    # "a2a":   supervisor calls the standalone research service over A2A.
    research_mode: Literal["local", "a2a"] = "local"

    # --- Python sandbox (analytics agent) ---
    sandbox_image: str = "orchestraos-sandbox:latest"
    sandbox_memory: str = "512m"
    # Only needed when the app itself runs in a container: the workspace's path on the host.
    sandbox_host_workspace: Path | None = None

    # --- Paths ---
    workspace_dir: Path = Path("workspace")  # the only folder agents may write to
    data_dir: Path = Path("data")  # checkpoints, vector store, etc.

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    def required_models(self) -> list[str]:
        """Every Ollama model the system needs, without duplicates, in a stable order."""
        return list(dict.fromkeys([self.small_model, self.large_model, self.embed_model]))


@lru_cache
def get_settings() -> Settings:
    """Settings are read once per process; tests override this via FastAPI dependencies."""
    return Settings()
