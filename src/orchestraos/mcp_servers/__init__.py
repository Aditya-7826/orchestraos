"""MCP servers (official `mcp` SDK, FastMCP). Each runs as its own process over stdio:

    python -m orchestraos.mcp_servers.search
    python -m orchestraos.mcp_servers.filesystem
    python -m orchestraos.mcp_servers.python_sandbox

They read their configuration from environment variables (see `env.py`), so they
can also be used on their own from any MCP client, e.g. Claude Desktop or the MCP Inspector.
"""
