"""MCP server: run Python (pandas/matplotlib) in a no-network Docker sandbox."""

from mcp.server.fastmcp import FastMCP

from orchestraos.mcp_servers import env
from orchestraos.tools.sandbox import DockerSandbox, SandboxResult


def build_server(sandbox: DockerSandbox | None = None) -> FastMCP:
    sandbox = sandbox or DockerSandbox(
        workspace=env.workspace_root(),
        image=env.sandbox_image(),
        memory=env.sandbox_memory(),
        host_workspace=env.sandbox_host_workspace(),
    )
    mcp = FastMCP(
        "python_sandbox",
        instructions=(
            "Run Python 3.12 with pandas, numpy and matplotlib. No network access. "
            "Workspace files are read-only under /data; save outputs (e.g. charts) "
            "to the current directory. Print the results you want to see."
        ),
        log_level="WARNING",
    )

    @mcp.tool()
    def run_python(code: str, timeout_seconds: int = 60) -> SandboxResult:
        """Run a Python script. Read inputs from /data/<file>, save charts with
        plt.savefig('chart.png'), and print() results. Returns stdout, stderr and saved files."""
        return sandbox.run_python(code, timeout_seconds=max(1, min(timeout_seconds, 120)))

    return mcp


if __name__ == "__main__":
    build_server().run("stdio")
