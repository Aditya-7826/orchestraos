"""Web search (DuckDuckGo, no API key) and page fetching.

`fetch_page` only fetches public http(s) URLs. Without that check, a prompt-
injected page could make the agent request http://localhost:... or a cloud
metadata address (SSRF). Redirects are followed by hand so each hop is checked.
"""

import ipaddress
import re
import socket
from collections.abc import Callable
from typing import TypedDict
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

MAX_REDIRECTS = 5
USER_AGENT = "OrchestraOS/0.1 (local research agent)"


class SearchHit(TypedDict):
    title: str
    url: str
    snippet: str


class Page(TypedDict):
    url: str
    title: str
    text: str


class UnsafeURLError(ValueError):
    pass


Resolver = Callable[[str], list[str]]


def resolve_host(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def check_public_url(url: str, resolve: Resolver = resolve_host) -> None:
    """Raise UnsafeURLError unless `url` is http(s) and every address it resolves to is public."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise UnsafeURLError(f"only http(s) URLs can be fetched: {url}")
    try:
        addresses = resolve(parts.hostname)
    except OSError as exc:
        raise UnsafeURLError(f"cannot resolve {parts.hostname}: {exc}") from exc
    for address in addresses:
        if not ipaddress.ip_address(address.split("%")[0]).is_global:
            raise UnsafeURLError(f"refusing non-public address {address} for {parts.hostname}")


def web_search(query: str, max_results: int = 5, ddgs: DDGS | None = None) -> list[SearchHit]:
    results = (ddgs or DDGS()).text(query, max_results=max_results)
    return [
        SearchHit(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
        for r in results
        if r.get("href")
    ]


def html_to_text(html: str) -> tuple[str, str]:
    """(title, readable text) with scripts, styles and page chrome removed."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(
        ["title", "script", "style", "noscript", "nav", "header", "footer", "aside", "form"]
    ):
        tag.decompose()
    text = soup.get_text("\n")
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    return title, "\n".join(line for line in lines if line)


def fetch_page(
    url: str,
    max_chars: int = 6000,
    client: httpx.Client | None = None,
    resolve: Resolver = resolve_host,
) -> Page:
    own_client = client is None
    client = client or httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT})
    try:
        for _ in range(MAX_REDIRECTS + 1):
            check_public_url(url, resolve)
            response = client.get(url, follow_redirects=False)
            if response.is_redirect and "location" in response.headers:
                url = urljoin(url, response.headers["location"])
                continue
            response.raise_for_status()
            break
        else:
            raise UnsafeURLError(f"too many redirects for {url}")
    finally:
        if own_client:
            client.close()

    content_type = response.headers.get("content-type", "")
    if "html" in content_type:
        title, text = html_to_text(response.text)
    elif content_type.startswith("text/") or "json" in content_type:
        title, text = "", response.text
    else:
        raise ValueError(f"unsupported content type {content_type!r} at {url}")

    if len(text) > max_chars:
        text = text[:max_chars] + "\n[truncated]"
    return Page(url=url, title=title, text=text)
