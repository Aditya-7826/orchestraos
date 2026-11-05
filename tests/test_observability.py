import importlib.util
from pathlib import Path

from fakes import fake_router

from orchestraos.agents.base import build_stub_agent
from orchestraos.graph.state import AGENT_NAMES
from orchestraos.graph.supervisor import build_supervisor
from orchestraos.memory.base import NullMemory
from orchestraos.observability import session_metadata, traced, tracing_enabled


def graph():
    agents = {name: build_stub_agent(name) for name in AGENT_NAMES}
    return build_supervisor(fake_router(lambda m: "{}"), agents, NullMemory())


def test_tracing_is_off_by_default(settings):
    assert not tracing_enabled(settings)
    g = graph()
    assert traced(g, settings) is g  # untouched: no callbacks, nothing sent anywhere


def test_keys_alone_dont_turn_tracing_on(settings):
    settings = settings.model_copy(
        update={"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}
    )
    assert not tracing_enabled(settings)


def test_tracing_attaches_a_langfuse_handler(settings):
    settings = settings.model_copy(
        update={
            "tracing": True,
            "langfuse_public_key": "pk-lf-test",
            "langfuse_secret_key": "sk-lf-test",
            "langfuse_base_url": "http://127.0.0.1:9",  # never contacted in this test
        }
    )
    g = traced(graph(), settings)
    (handler,) = g.config["callbacks"]
    assert type(handler).__module__.startswith("langfuse")
    assert g.checkpointer is None  # same graph otherwise


def test_runs_are_grouped_by_thread():
    assert session_metadata("abc") == {"langfuse_session_id": "abc"}


# --- make setup -------------------------------------------------------------


def load_setup_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "setup_env.py"
    spec = importlib.util.spec_from_file_location("setup_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_generates_secrets_once_and_keeps_existing_values(tmp_path, monkeypatch):
    setup = load_setup_script()
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "sales.csv").write_text("a\n1\n")
    (tmp_path / ".env.example").write_text("SMALL_MODEL=x\nLANGFUSE_SALT=mine\n")
    monkeypatch.setattr(setup, "ROOT", tmp_path)

    setup.main()
    first = (tmp_path / ".env").read_text()
    setup.main()
    assert (tmp_path / ".env").read_text() == first  # idempotent

    values = dict(line.split("=", 1) for line in first.splitlines() if "=" in line)
    assert values["LANGFUSE_SALT"] == "mine"  # never overwritten
    assert len(values["LANGFUSE_ENCRYPTION_KEY"]) == 64
    assert values["LANGFUSE_PUBLIC_KEY"].startswith("pk-lf-")
    assert (tmp_path / "workspace" / "sales.csv").exists()
