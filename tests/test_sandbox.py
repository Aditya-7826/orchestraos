import shutil
import subprocess

import pytest

from orchestraos.tools.sandbox import DockerSandbox

IMAGE = "orchestraos-sandbox:latest"
# Generous: on an 8 GB laptop with a model loaded, starting a container alone can take 10s+.
SLOW_MACHINE_TIMEOUT = 120


def flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_command_locks_the_container_down(tmp_path):
    cmd = DockerSandbox(workspace=tmp_path, memory="256m").command("abc")
    assert flag_value(cmd, "--network") == "none"
    assert flag_value(cmd, "--memory") == flag_value(cmd, "--memory-swap") == "256m"
    assert flag_value(cmd, "--cap-drop") == "ALL"
    assert "--read-only" in cmd
    mounts = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--mount"]
    assert mounts[0].endswith("dst=/data,readonly")
    assert mounts[1].endswith("dst=/output") and "outputs" in mounts[1]


def test_command_uses_host_path_when_given(tmp_path):
    cmd = DockerSandbox(workspace=tmp_path, host_workspace=tmp_path / "host").command("abc")
    assert str(tmp_path / "host") in flag_value(cmd, "--mount")


def test_timeout_kills_the_container(tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)

    result = DockerSandbox(workspace=tmp_path, run=fake_run).run_python("while True: pass", 1)
    assert result["timed_out"] and result["exit_code"] is None
    assert calls[1][:2] == ["docker", "kill"]


# --- real Docker ------------------------------------------------------------


def sandbox_available() -> bool:
    if not shutil.which("docker"):
        return False
    probe = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return probe.returncode == 0


needs_docker = pytest.mark.skipif(
    not sandbox_available(), reason=f"needs Docker and the {IMAGE} image (make sandbox-image)"
)


@needs_docker
def test_runs_pandas_and_saves_a_chart(tmp_path):
    (tmp_path / "sales.csv").write_text("month,sales\n1,10\n2,30\n")
    code = """
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('/data/sales.csv')
print(df['sales'].sum())
df.plot(x='month', y='sales'); plt.savefig('chart.png')
"""
    result = DockerSandbox(workspace=tmp_path).run_python(code, SLOW_MACHINE_TIMEOUT)
    assert result["exit_code"] == 0, result["stderr"]
    assert result["stdout"].strip() == "40"
    assert len(result["files"]) == 1 and result["files"][0].endswith("/chart.png")
    assert (tmp_path / result["files"][0]).stat().st_size > 0


@needs_docker
def test_has_no_network(tmp_path):
    code = "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"
    result = DockerSandbox(workspace=tmp_path).run_python(code, SLOW_MACHINE_TIMEOUT)
    assert result["exit_code"] != 0
    assert "OSError" in result["stderr"] or "unreachable" in result["stderr"].lower()


@needs_docker
def test_workspace_is_read_only(tmp_path):
    result = DockerSandbox(workspace=tmp_path).run_python(
        "open('/data/x.txt', 'w')", SLOW_MACHINE_TIMEOUT
    )
    assert result["exit_code"] != 0 and "Read-only file system" in result["stderr"]
