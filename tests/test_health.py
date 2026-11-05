import pytest
from fastapi.testclient import TestClient

from orchestraos.api import main
from orchestraos.config import get_settings
from orchestraos.ollama import OllamaUnavailable


@pytest.fixture
def client(settings):
    main.app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_ok_when_all_models_pulled(client, monkeypatch):
    monkeypatch.setattr(
        main, "list_local_models", lambda url: {"small:3b", "large:7b", "embed:latest"}
    )
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ollama"] == {"reachable": True, "missing_models": [], "error": None}


def test_degraded_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(main, "list_local_models", lambda url: {"small:3b"})
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"]["missing_models"] == ["large:7b", "embed"]


def test_degraded_when_ollama_down(client, monkeypatch):
    def down(url):
        raise OllamaUnavailable("cannot reach Ollama")

    monkeypatch.setattr(main, "list_local_models", down)
    response = client.get("/health")
    assert response.status_code == 200  # the API itself is alive
    assert response.json()["ollama"]["reachable"] is False
