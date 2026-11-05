"""`make setup` helper: create .env and the workspace folder. Safe to run again.

- copies .env.example to .env if there's no .env yet
- sets HOST_WORKSPACE (the workspace's absolute path on this machine), which the
  dockerised API needs to mount files into sandbox containers
- creates workspace/ and adds the example data the first time
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    env = ROOT / ".env"
    if not env.exists():
        shutil.copy(ROOT / ".env.example", env)
        print("created .env from .env.example")

    workspace = ROOT / "workspace"
    lines = env.read_text(encoding="utf-8").splitlines()
    if not any(line.startswith("HOST_WORKSPACE=") for line in lines):
        # Forward slashes work for Docker on Windows, macOS and Linux.
        lines.append(f"HOST_WORKSPACE={workspace.as_posix()}")
        env.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"set HOST_WORKSPACE={workspace.as_posix()}")

    if not workspace.exists():
        workspace.mkdir()
        for example in (ROOT / "examples").iterdir():
            shutil.copy(example, workspace / example.name)
        print("created workspace/ with the example files")


if __name__ == "__main__":
    main()
