import httpx
import pytest

from orchestraos.tools import web


def public(host: str) -> list[str]:
    return ["93.184.215.14"]


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com", "javascript:alert(1)", "http://"],
)
def test_rejects_non_http_urls(url):
    with pytest.raises(web.UnsafeURLError):
        web.check_public_url(url, resolve=public)


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1"]
)
def test_rejects_private_addresses(address):
    with pytest.raises(web.UnsafeURLError, match="non-public"):
        web.check_public_url("http://sneaky.example", resolve=lambda host: [address])


def test_accepts_public_address():
    web.check_public_url("https://example.com/page", resolve=public)


def test_html_to_text_strips_chrome_and_scripts():
    html = """<html><head><title> My Page </title><script>evil()</script></head>
    <body><nav>Home | About</nav><h1>Heading</h1><p>Body   text.</p>
    <footer>(c) 2026</footer></body></html>"""
    title, text = web.html_to_text(html)
    assert title == "My Page"
    assert text == "Heading\nBody text."


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_page_returns_readable_text():
    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<title>T</title><p>Hello world</p>",
        )

    page = web.fetch_page("https://example.com", client=mock_client(handler), resolve=public)
    assert page == {"url": "https://example.com", "title": "T", "text": "Hello world"}


def test_fetch_page_checks_every_redirect_hop():
    def handler(request):
        return httpx.Response(302, headers={"location": "http://internal.example/admin"})

    def resolve(host):
        return ["10.0.0.1"] if host == "internal.example" else ["93.184.215.14"]

    with pytest.raises(web.UnsafeURLError, match="non-public"):
        web.fetch_page("https://example.com", client=mock_client(handler), resolve=resolve)


def test_fetch_page_rejects_binary_content():
    def handler(request):
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG")

    with pytest.raises(ValueError, match="unsupported content type"):
        web.fetch_page("https://example.com/x.png", client=mock_client(handler), resolve=public)


def test_web_search_maps_ddgs_results():
    class FakeDDGS:
        def text(self, query, max_results):
            assert (query, max_results) == ("langgraph", 2)
            return [
                {"title": "LangGraph", "href": "https://a.example", "body": "graphs"},
                {"title": "no url", "href": "", "body": "dropped"},
            ]

    hits = web.web_search("langgraph", max_results=2, ddgs=FakeDDGS())
    assert hits == [{"title": "LangGraph", "url": "https://a.example", "snippet": "graphs"}]
