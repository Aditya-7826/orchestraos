import pytest

from orchestraos.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings that never read the developer's real .env."""
    return Settings(
        _env_file=None,
        ollama_base_url="http://ollama.test",
        small_model="small:3b",
        large_model="large:7b",
        embed_model="embed",
    )
