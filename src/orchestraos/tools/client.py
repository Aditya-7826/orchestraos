"""Starts the MCP servers and turns their tools into LangChain tools for the agents.

Each server is launched once as a subprocess (stdio transport) and kept alive for
as long as the `open_toolset` context is open, instead of a new process per call.
"""

import os
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_mcp_adapters.tools import load_mcp_tools

from orchestraos.config import Settings

SERVER_MODULES = {
    "search": "orchestraos.mcp_servers.search",
    "filesystem": "orchestraos.mcp_servers.filesystem",
    "python_sandbox": "orchestraos.mcp_servers.python_sandbox",
}


class ToolCallError(RuntimeError):
    """The tool ran but reported an error (bad path, blocked URL, search failure, ...)."""


class ToolCaller(Protocol):
    """All an agent needs from the tool layer. Tests pass a fake; the app passes a Toolset."""

    async def call(self, name: str, **args: Any) -> Any: ...


@dataclass
class Toolset:
    by_server: dict[str, list[BaseTool]]

    async def call(self, name: str, **args: Any) -> Any:
        """Call a tool and return its structured result, or raise ToolCallError."""
        (tool,) = self.get(name)
        call = {"type": "tool_call", "id": uuid.uuid4().hex, "name": name, "args": args}
        message = await tool.ainvoke(call)
        if message.status == "error":
            raise ToolCallError(_text(message.content))
        structured = (message.artifact or {}).get("structured_content")
        if structured is None:
            return _text(message.content)
        # FastMCP wraps non-object return values (lists, strings) as {"result": value}.
        return structured["result"] if set(structured) == {"result"} else structured

    def get(self, *names: str) -> list[BaseTool]:
        """Tools by name, e.g. toolset.get("web_search", "fetch_page")."""
        tools = {t.name: t for server_tools in self.by_server.values() for t in server_tools}
        missing = [n for n in names if n not in tools]
        if missing:
            raise KeyError(f"unknown tools: {missing}; available: {sorted(tools)}")
        return [tools[n] for n in names]


@dataclass
class RestrictedTools:
    """A view of a ToolCaller that only allows some tools (least privilege per agent)."""

    inner: ToolCaller
    allowed: frozenset[str]

    async def call(self, name: str, **args: Any) -> Any:
        if name not in self.allowed:
            raise PermissionError(
                f"tool {name!r} is not allowed here; allowed: {sorted(self.allowed)}"
            )
        return await self.inner.call(name, **args)


def restrict(tools: ToolCaller, *names: str) -> RestrictedTools:
    return RestrictedTools(tools, frozenset(names))


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))


def server_connections(
    settings: Settings, servers: list[str] | None = None
) -> dict[str, StdioConnection]:
    # The subprocess gets only the env we pass, so start from ours (PATH, SYSTEMROOT, ...).
    env = {
        **os.environ,
        "WORKSPACE_DIR": str(settings.workspace_dir.resolve()),
        "SANDBOX_IMAGE": settings.sandbox_image,
        "SANDBOX_MEMORY": settings.sandbox_memory,
    }
    if settings.sandbox_host_workspace:
        env["SANDBOX_HOST_WORKSPACE"] = str(settings.sandbox_host_workspace)
    return {
        name: StdioConnection(
            transport="stdio", command=sys.executable, args=["-m", module], env=env
        )
        for name, module in SERVER_MODULES.items()
        if servers is None or name in servers
    }


@asynccontextmanager
async def open_toolset(
    settings: Settings, servers: list[str] | None = None
) -> AsyncIterator[Toolset]:
    client = MultiServerMCPClient(server_connections(settings, servers))
    async with AsyncExitStack() as stack:
        by_server = {}
        for name in client.connections:
            session = await stack.enter_async_context(client.session(name))
            by_server[name] = await load_mcp_tools(session, server_name=name)
        yield Toolset(by_server)
