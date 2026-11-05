"""File access confined to one folder (the workspace).

Every path from an agent is treated as untrusted: it is resolved (following
`..` and symlinks) and rejected unless the result is still inside the root.
"""

from pathlib import Path

from pypdf import PdfReader

MAX_LISTED_FILES = 200
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".py", ".html", ".yaml", ".yml", ".tsv", ".log"}


class WorkspaceError(ValueError):
    """A path outside the workspace, a missing file, or an unreadable format."""


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        """Absolute path for `relative`, guaranteed to be inside the workspace."""
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise WorkspaceError(f"path is outside the workspace: {relative}")
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def list_files(self, subdir: str = ".") -> list[str]:
        base = self.resolve(subdir)
        if not base.is_dir():
            raise WorkspaceError(f"not a directory: {subdir}")
        files = sorted(
            self.relative(p)
            for p in base.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(base).parts)
        )
        return files[:MAX_LISTED_FILES]

    def read_file(self, path: str, max_chars: int = 20_000) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"no such file: {path}")

        suffix = target.suffix.lower()
        if suffix == ".pdf":
            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(target).pages)
        elif suffix in TEXT_SUFFIXES:
            text = target.read_text(encoding="utf-8", errors="replace")
        else:
            raise WorkspaceError(f"unsupported file type {suffix!r}; can read text and PDF files")

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[truncated: showing {max_chars} of {len(text)} chars]"
        return text

    def write_file(self, path: str, content: str) -> str:
        target = self.resolve(path)
        if target == self.root or target.is_dir():
            raise WorkspaceError(f"not a file path: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return self.relative(target)
