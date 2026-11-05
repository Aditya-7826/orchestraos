import pytest

from orchestraos.tools.workspace import Workspace, WorkspaceError


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "workspace")


@pytest.mark.parametrize("path", ["../secret.txt", "a/../../secret.txt", "/etc/passwd", "C:/x"])
def test_rejects_paths_outside_workspace(ws, path):
    with pytest.raises(WorkspaceError):
        ws.write_file(path, "x")


def test_rejects_symlink_escape(ws, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    try:
        (ws.root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks needs extra privileges on this OS")
    with pytest.raises(WorkspaceError):
        ws.read_file("link/secret.txt")


def test_write_then_read_and_list(ws):
    assert ws.write_file("reports/q1.md", "# Q1") == "reports/q1.md"
    assert ws.read_file("reports/q1.md") == "# Q1"
    assert ws.list_files() == ["reports/q1.md"]


def test_list_skips_hidden_files(ws):
    ws.write_file("visible.txt", "v")
    ws.write_file(".hidden/secret.txt", "h")
    assert ws.list_files() == ["visible.txt"]


def test_read_truncates_long_files(ws):
    ws.write_file("big.txt", "x" * 100)
    text = ws.read_file("big.txt", max_chars=10)
    assert text.startswith("x" * 10) and "truncated" in text


def test_read_rejects_binary_formats(ws):
    (ws.root / "image.png").write_bytes(b"\x89PNG")
    with pytest.raises(WorkspaceError, match="unsupported"):
        ws.read_file("image.png")


def test_read_missing_file(ws):
    with pytest.raises(WorkspaceError, match="no such file"):
        ws.read_file("nope.txt")


def test_cannot_overwrite_a_directory(ws):
    ws.write_file("dir/file.txt", "x")
    with pytest.raises(WorkspaceError):
        ws.write_file("dir", "x")
