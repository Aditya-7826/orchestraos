"""The guided tour: runs real graph code with fake models, scripted answers and no Docker."""

from contextlib import asynccontextmanager

from test_action import llm, supervisor, workspace_tools

from orchestraos import demo
from orchestraos.doctor import Check

SAVE_GOAL_PLAN = {"actions": [{"path": "notes/demo-checklist.md", "from_step": 1}]}


class ScriptedConsole(demo.Console):
    def __init__(self, answers: list[str], auto: bool = False):
        self.lines: list[str] = []
        self.prompts: list[str] = []
        queue = list(answers)

        def ask(prompt: str) -> str:
            self.prompts.append(prompt)
            return queue.pop(0) if queue else ""

        super().__init__(auto=auto, ask=ask, show=self.lines.append)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def patch_app(monkeypatch, tools):
    @asynccontextmanager
    async def fake_open_app(settings):
        yield supervisor(llm(SAVE_GOAL_PLAN), tools)

    monkeypatch.setattr(demo, "open_app", fake_open_app)
    monkeypatch.setattr(demo, "run_checks", lambda settings: [Check("Ollama", True, "ok")])


async def test_approval_step_asks_and_writes_on_yes(settings, monkeypatch, tmp_path):
    tools, written = workspace_tools()
    patch_app(monkeypatch, tools)
    settings = settings.model_copy(update={"workspace_dir": tmp_path, "data_dir": tmp_path / "d"})
    console = ScriptedConsole(answers=["", "y"])  # Enter to start the step, then approve

    assert await demo.run_demo(settings, console, only=[4]) == 0
    assert console.prompts[1].strip() == "Approve? [y/N]"
    assert "The run is paused. It wants to make these changes:" in console.text
    assert "write_file -> notes/demo-checklist.md" in console.text
    assert written  # approved, so the action agent wrote it
    assert "Saved to notes/demo-checklist.md" in console.text


async def test_rejecting_writes_nothing(settings, monkeypatch, tmp_path):
    tools, written = workspace_tools()
    patch_app(monkeypatch, tools)
    settings = settings.model_copy(update={"workspace_dir": tmp_path, "data_dir": tmp_path / "d"})
    console = ScriptedConsole(answers=["", "n"])

    await demo.run_demo(settings, console, only=[4])
    assert written == {}
    assert "Not saved" in console.text


async def test_yes_mode_never_prompts(settings, monkeypatch, tmp_path):
    tools, _ = workspace_tools()
    patch_app(monkeypatch, tools)
    settings = settings.model_copy(update={"workspace_dir": tmp_path, "data_dir": tmp_path / "d"})
    console = ScriptedConsole(answers=[], auto=True)

    assert await demo.run_demo(settings, console) == 0
    assert console.prompts == []
    for n, step in enumerate(demo.STEPS, 1):
        assert f"Step {n}/{len(demo.STEPS)}: {step.title}" in console.text
    assert "triage (small model)" in console.text and "plan (large model):" in console.text
    assert (tmp_path / "sales.csv").exists()  # sample data copied in for the analytics step


async def test_failed_setup_check_stops_the_tour(settings, monkeypatch, tmp_path):
    tools, _ = workspace_tools()
    patch_app(monkeypatch, tools)
    monkeypatch.setattr(
        demo, "run_checks",
        lambda s: [Check("model qwen2.5:7b", False, "not pulled", hint="ollama pull qwen2.5:7b")],
    )  # fmt: skip
    console = ScriptedConsole(answers=[], auto=True)
    assert await demo.run_demo(settings, console) == 1
    assert "ollama pull qwen2.5:7b" in console.text
    assert "Step 2/" not in console.text


def test_demo_uses_separate_memory(settings):
    isolated = demo.demo_settings(settings.model_copy(update={"neo4j_uri": "bolt://x"}))
    assert isolated.data_dir == settings.data_dir / "demo"
    assert isolated.neo4j_uri is None and not isolated.tracing
    assert isolated.workspace_dir == settings.workspace_dir  # files land where you can open them


def test_describe_events():
    assert demo.describe({"type": "triage", "mode": "direct"}) == [
        "  triage (small model): direct answer"
    ]
    step = {"type": "step", "step_id": 2, "agent": "analytics", "ok": False,
            "preview": "Error:\n  boom"}  # fmt: skip
    assert demo.describe(step) == ["  step 2 analytics: FAILED - Error: boom"]
