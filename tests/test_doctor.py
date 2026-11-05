import subprocess

from orchestraos import doctor
from orchestraos.ollama import OllamaUnavailable


def test_ollama_down_is_single_failure(settings):
    def down(url):
        raise OllamaUnavailable("refused")

    checks = doctor.check_ollama(settings, list_models=down)
    assert len(checks) == 1 and not checks[0].ok


def test_missing_model_gets_pull_hint(settings):
    checks = doctor.check_ollama(settings, list_models=lambda url: {"small:3b", "embed:latest"})
    failed = [c for c in checks if not c.ok]
    assert [c.name for c in failed] == ["model large:7b"]
    assert failed[0].hint == "ollama pull large:7b"


def test_docker_not_installed():
    def missing(*args, **kwargs):
        raise FileNotFoundError

    assert not doctor.check_docker(run=missing).ok


def test_docker_daemon_stopped():
    def stopped(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="")

    assert doctor.check_docker(run=stopped).detail == "daemon not running"


def test_docker_running():
    def running(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="29.0\n", stderr="")

    assert doctor.check_docker(run=running).ok


def test_low_ram_is_warning_not_failure():
    check = doctor.check_memory(8 * doctor.GIB)
    assert not check.ok and check.warn_only


def test_enough_ram_passes():
    assert doctor.check_memory(16 * doctor.GIB).ok
