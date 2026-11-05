"""Runs untrusted Python (written by the analytics agent) in a locked-down container.

The container gets: no network, capped memory/CPU/processes, a read-only root
filesystem, no Linux capabilities, the workspace mounted read-only at /data,
and a fresh writable folder at /output for charts and other results. That
folder is `workspace/outputs/<run id>/` on the host.
"""

import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

MAX_OUTPUT_CHARS = 10_000


class SandboxResult(TypedDict):
    exit_code: int | None  # None if the run timed out
    timed_out: bool
    stdout: str
    stderr: str
    files: list[str]  # workspace-relative paths of files written to /output


@dataclass
class DockerSandbox:
    workspace: Path
    image: str = "orchestraos-sandbox:latest"
    memory: str = "512m"
    cpus: str = "1"
    # Host path of the workspace, for when this code itself runs inside a container
    # and asks the host's Docker daemon to start the sandbox (paths must be host paths).
    host_workspace: Path | None = None
    run: Callable[..., subprocess.CompletedProcess] = field(default=subprocess.run, repr=False)

    def command(self, run_id: str) -> list[str]:
        host_root = self.host_workspace or self.workspace.resolve()
        return [
            "docker", "run", "--rm", "-i",
            "--name", f"orchestraos-sandbox-{run_id}",
            "--network", "none",
            "--memory", self.memory, "--memory-swap", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", "128",
            "--read-only",
            "--tmpfs", "/tmp:size=64m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--env", "MPLBACKEND=Agg",
            "--mount", f"type=bind,src={host_root},dst=/data,readonly",
            "--mount", f"type=bind,src={host_root / 'outputs' / run_id},dst=/output",
            "--workdir", "/output",
            self.image,
            "python", "-",
        ]  # fmt: skip

    def run_python(self, code: str, timeout_seconds: int = 60) -> SandboxResult:
        run_id = uuid.uuid4().hex[:10]
        output_dir = self.workspace.resolve() / "outputs" / run_id
        output_dir.mkdir(parents=True)
        # The sandbox runs as an unprivileged user (uid 1000) but this folder may be created
        # by root (e.g. when the API itself runs in a container), so open it up. It's a
        # fresh, per-run folder that only this one sandbox container mounts.
        output_dir.chmod(0o777)

        try:
            proc = self.run(
                self.command(run_id),
                input=code,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            # The CLI timing out doesn't stop the container; kill it explicitly.
            self.run(["docker", "kill", f"orchestraos-sandbox-{run_id}"], capture_output=True)
            return SandboxResult(
                exit_code=None,
                timed_out=True,
                stdout="",
                stderr=f"timed out after {timeout_seconds}s",
                files=self._outputs(output_dir),
            )

        return SandboxResult(
            exit_code=proc.returncode,
            timed_out=False,
            stdout=_clip(proc.stdout),
            stderr=_clip(proc.stderr),
            files=self._outputs(output_dir),
        )

    def _outputs(self, output_dir: Path) -> list[str]:
        root = self.workspace.resolve()
        return sorted(p.relative_to(root).as_posix() for p in output_dir.rglob("*") if p.is_file())


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n[truncated {len(text) - MAX_OUTPUT_CHARS} chars]"
