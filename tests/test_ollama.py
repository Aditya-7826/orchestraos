import httpx
import pytest

from orchestraos.ollama import (
    OllamaUnavailable,
    list_local_models,
    missing_models,
    normalize_model_name,
)


def client_returning(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_normalize_adds_latest_tag():
    assert normalize_model_name("nomic-embed-text") == "nomic-embed-text:latest"
    assert normalize_model_name("qwen2.5:3b") == "qwen2.5:3b"


def test_list_local_models_parses_tags():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}, {"name": "embed"}]})

    models = list_local_models("http://ollama.test/", client=client_returning(handler))
    assert models == {"qwen2.5:3b", "embed:latest"}


def test_list_local_models_wraps_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaUnavailable):
        list_local_models("http://ollama.test", client=client_returning(handler))


def test_list_local_models_wraps_http_errors():
    with pytest.raises(OllamaUnavailable):
        list_local_models(
            "http://ollama.test", client=client_returning(lambda r: httpx.Response(500))
        )


def test_missing_models_matches_untagged_names():
    available = {"qwen2.5:3b", "embed:latest"}
    assert missing_models(["qwen2.5:3b", "embed", "qwen2.5:7b"], available) == ["qwen2.5:7b"]
