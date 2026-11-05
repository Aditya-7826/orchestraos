"""Starts the MCP servers and turns their tools into LangChain tools for the agents.

Each server is launched once as a subprocess (stdio transport) and kept alive for
as long as the `open_toolset` context is open, instead of a new process per call.
"""

import os
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

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


@dataclass
class Toolset:
    by_server: dict[str, list[BaseTool]]

    def get(self, *names: str) -> list[BaseTool]:
        """Tools by name, e.g. toolset.get("web_search", "fetch_page")."""
        tools = {t.name: t for server_tools in self.by_server.values() for t in server_tools}
        missing = [n for n in names if n not in tools]
        if missing:
            raise KeyError(f"unknown tools: {missing}; available: {sorted(tools)}")
        return [tools[n] for n in names]


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
