"""Guards against text files being saved with the wrong encoding (UTF-8 read as cp1252)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".csv"}
SKIP = {".venv", ".git", "workspace", "data", ".pytest_cache", ".ruff_cache"}
# What common characters (— – ✅ →) look like after being decoded as cp1252.
MOJIBAKE = ("â€", "Ã¢", "âœ")


def text_files():
    for path in ROOT.rglob("*"):
        if path.suffix in SUFFIXES and not SKIP.intersection(path.relative_to(ROOT).parts):
            yield path


def test_no_mojibake_in_repo():
    bad = [
        str(path.relative_to(ROOT))
        for path in text_files()
        if any(marker in path.read_text(encoding="utf-8") for marker in MOJIBAKE)
    ]
    assert bad == []
