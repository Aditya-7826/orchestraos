"""`orchestraos-doctor`: checks that this machine is ready to run OrchestraOS.

Each check is a small pure-ish function returning a `Check`, so they're easy to
unit test by passing in fakes instead of touching the real Ollama/Docker.
"""

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

import psutil

from orchestraos.config import Settings
from orchestraos.ollama import OllamaUnavailable, list_local_models, missing_models

GIB = 1024**3
# Below this much RAM, running every optional profile at once is likely to swap or hang.
FULL_STACK_RAM_GIB = 12


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""
    warn_only: bool = False  # a failed warning doesn't fail the doctor run


def check_ollama(
    settings: Settings, list_models: Callable[[str], set[str]] = list_local_models
) -> list[Check]:
    try:
        available = list_models(settings.ollama_base_url)
    except OllamaUnavailable as exc:
        return [
            Check(
                "Ollama server",
                ok=False,
                detail=str(exc),
                hint="Install Ollama from https://ollama.com and make sure it is running.",
            )
        ]

    checks = [Check("Ollama server", ok=True, detail=settings.ollama_base_url)]
    missing = set(missing_models(settings.required_models(), available))
    for model in settings.required_models():
        if model in missing:
            checks.append(
                Check(f"model {model}", ok=False, detail="not pulled", hint=f"ollama pull {model}")
            )
        else:
            checks.append(Check(f"model {model}", ok=True, detail="pulled"))
    return checks


def check_docker(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Check:
    try:
        result = run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError:
        return Check(
            "Docker",
            ok=False,
            detail="docker CLI not found",
            hint="Install Docker Desktop (free for personal use) or Docker Engine.",
        )
    except subprocess.TimeoutExpired:
        return Check(
            "Docker", ok=False, detail="`docker info` timed out", hint="Restart Docker Desktop."
        )

    if result.returncode != 0:
        return Check("Docker", ok=False, detail="daemon not running", hint="Start Docker Desktop.")
    return Check("Docker", ok=True, detail=f"engine {result.stdout.strip()}")


def check_sandbox_image(
    image: str, run: Callable[..., subprocess.CompletedProcess] = subprocess.run
) -> Check:
    try:
        result = run(["docker", "image", "inspect", image], capture_output=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return Check(f"sandbox image {image}", ok=False, detail="docker unavailable")
    if result.returncode != 0:
        return Check(
            f"sandbox image {image}", ok=False, detail="not built", hint="make sandbox-image"
        )
    return Check(f"sandbox image {image}", ok=True, detail="built")


def check_memory(total_bytes: int) -> Check:
    total_gib = total_bytes / GIB
    if total_gib < FULL_STACK_RAM_GIB:
        return Check(
            "RAM",
            ok=False,
            warn_only=True,
            detail=f"{total_gib:.1f} GiB",
            hint="Run the core profile only; enable the optional profiles one at a time.",
        )
    return Check("RAM", ok=True, detail=f"{total_gib:.1f} GiB")


def run_checks(settings: Settings) -> list[Check]:
    return [
        *check_ollama(settings),
        check_docker(),
        check_sandbox_image(settings.sandbox_image),
        check_memory(psutil.virtual_memory().total),
    ]


def format_check(check: Check) -> str:
    label = "OK  " if check.ok else ("WARN" if check.warn_only else "FAIL")
    line = f"[{label}] {check.name}: {check.detail}"
    if not check.ok and check.hint:
        line += f"\n       -> {check.hint}"
    return line


def main() -> int:
    try:
        settings = Settings()
    except Exception as exc:  # pydantic ValidationError, with a readable message
        print(f"[FAIL] config: {exc}\n       -> Copy .env.example to .env and fill it in.")
        return 1

    checks = run_checks(settings)
    for check in checks:
        print(format_check(check))

    failed = [c for c in checks if not c.ok and not c.warn_only]
    print("\nAll required checks passed." if not failed else f"\n{len(failed)} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
