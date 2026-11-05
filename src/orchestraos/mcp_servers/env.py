"""Environment variables the MCP servers read. The tool client sets them when spawning servers."""

import os
from pathlib import Path


def workspace_root() -> Path:
    return Path(os.environ.get("WORKSPACE_DIR", "workspace"))


def sandbox_image() -> str:
    return os.environ.get("SANDBOX_IMAGE", "orchestraos-sandbox:latest")


def sandbox_memory() -> str:
    return os.environ.get("SANDBOX_MEMORY", "512m")


def sandbox_host_workspace() -> Path | None:
    value = os.environ.get("SANDBOX_HOST_WORKSPACE")
    return Path(value) if value else None
