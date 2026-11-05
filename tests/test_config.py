import pytest
from pydantic import ValidationError

from orchestraos.config import Settings

MODELS = {"SMALL_MODEL": "small:3b", "LARGE_MODEL": "large:7b", "EMBED_MODEL": "embed"}


@pytest.fixture
def model_env(monkeypatch):
    for key, value in MODELS.items():
        monkeypatch.setenv(key, value)


def test_loads_models_from_env(model_env):
    settings = Settings(_env_file=None)
    assert settings.small_model == "small:3b"
    assert settings.large_model == "large:7b"
    assert settings.research_mode == "local"


def test_model_names_are_required(monkeypatch):
    for key in MODELS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rejects_unknown_research_mode(model_env, monkeypatch):
    monkeypatch.setenv("RESEARCH_MODE", "carrier-pigeon")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_required_models_deduplicates(model_env, monkeypatch):
    monkeypatch.setenv("LARGE_MODEL", "small:3b")  # one model doing both jobs
    assert Settings(_env_file=None).required_models() == ["small:3b", "embed"]
