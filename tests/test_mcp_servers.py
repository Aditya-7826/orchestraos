"""The MCP layer: servers expose the right tools, and the client can drive them over stdio."""

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from orchestraos.mcp_servers import filesystem, python_sandbox, search
from orchestraos.tools.client import ToolCallError, open_toolset, server_connections
from orchestraos.tools.sandbox import DockerSandbox


async def tool_names(server) -> set[str]:
    return {t.name for t in await server.list_tools()}


async def test_each_server_exposes_its_tools(tmp_path):
    assert await tool_names(search.build_server()) == {"web_search", "fetch_page"}
    assert await tool_names(filesystem.build_server(tmp_path)) == {
        "list_files", "read_file", "write_file",
    }  # fmt: skip
    sandbox = DockerSandbox(workspace=tmp_path)
    assert await tool_names(python_sandbox.build_server(sandbox)) == {"run_python"}


async def test_filesystem_server_enforces_workspace(tmp_path):
    server = filesystem.build_server(tmp_path)
    _, structured = await server.call_tool("write_file", {"path": "notes.md", "content": "hi"})
    assert structured == {"result": "notes.md"}
    with pytest.raises(ToolError, match="outside the workspace"):
        await server.call_tool("read_file", {"path": "../escape.txt"})


def test_connections_pass_workspace_to_servers(settings, tmp_path):
    settings = settings.model_copy(update={"workspace_dir": tmp_path})
    connections = server_connections(settings)
    assert set(connections) == {"search", "filesystem", "python_sandbox"}
    env = connections["filesystem"]["env"]
    assert env["WORKSPACE_DIR"] == str(tmp_path.resolve())
    assert "PATH" in env  # inherits our environment, not just the overrides


async def test_client_drives_filesystem_server_over_stdio(settings, tmp_path):
    """Spawns the real server process and calls it through the LangChain tool wrappers."""
    settings = settings.model_copy(update={"workspace_dir": tmp_path})
    async with open_toolset(settings, servers=["filesystem"]) as toolset:
        write, read = toolset.get("write_file", "read_file")
        await write.ainvoke({"path": "hello.txt", "content": "over MCP"})
        assert "over MCP" in str(await read.ainvoke({"path": "hello.txt"}))
    assert (tmp_path / "hello.txt").read_text() == "over MCP"


async def test_toolset_reports_unknown_tools(settings, tmp_path):
    settings = settings.model_copy(update={"workspace_dir": tmp_path})
    async with open_toolset(settings, servers=["filesystem"]) as toolset:
        with pytest.raises(KeyError, match="web_search"):
            toolset.get("web_search")


async def test_toolset_call_returns_structured_data_or_raises(settings, tmp_path):
    settings = settings.model_copy(update={"workspace_dir": tmp_path})
    async with open_toolset(settings, servers=["filesystem"]) as toolset:
        assert await toolset.call("write_file", path="a.txt", content="x") == "a.txt"
        assert await toolset.call("list_files") == ["a.txt"]
        with pytest.raises(ToolCallError, match="outside the workspace"):
            await toolset.call("read_file", path="../nope.txt")
