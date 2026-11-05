"""MCP server: read and write files, confined to the workspace folder."""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from orchestraos.mcp_servers.env import workspace_root
from orchestraos.tools.workspace import Workspace


def build_server(root: Path | None = None) -> FastMCP:
    workspace = Workspace(root or workspace_root())
    mcp = FastMCP(
        "filesystem",
        instructions="Files in the user's workspace folder. Paths are relative to that folder.",
        log_level="WARNING",
    )

    @mcp.tool()
    def list_files(subdir: str = ".") -> list[str]:
        """List files in the workspace (recursively), as paths relative to the workspace."""
        return workspace.list_files(subdir)

    @mcp.tool()
    def read_file(path: str, max_chars: int = 20_000) -> str:
        """Read a text or PDF file from the workspace."""
        return workspace.read_file(path, max_chars=max_chars)

    @mcp.tool()
    def write_file(path: str, content: str) -> str:
        """Write a text file in the workspace (creating folders as needed). Returns its path."""
        return workspace.write_file(path, content)

    return mcp


if __name__ == "__main__":
    build_server().run("stdio")
