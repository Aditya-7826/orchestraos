"""MCP server: web search and page fetching."""

from mcp.server.fastmcp import FastMCP

from orchestraos.tools import web


def build_server() -> FastMCP:
    mcp = FastMCP(
        "search",
        instructions="Search the web with DuckDuckGo and read web pages.",
        log_level="WARNING",
    )

    @mcp.tool()
    def web_search(query: str, max_results: int = 5) -> list[web.SearchHit]:
        """Search the web. Returns results with title, url and a short snippet."""
        return web.web_search(query, max_results=min(max_results, 10))

    @mcp.tool()
    def fetch_page(url: str, max_chars: int = 6000) -> web.Page:
        """Fetch a public web page and return its title and readable text."""
        return web.fetch_page(url, max_chars=min(max_chars, 20_000))

    return mcp


if __name__ == "__main__":
    build_server().run("stdio")
