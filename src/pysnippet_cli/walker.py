"""Recursive directory walker that yields source files worth indexing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

# Directories we never want to descend into, regardless of project type.
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".nox",
        ".venv",
        "venv",
        "env",
        ".env",
        "node_modules",
        "bower_components",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".output",
        "target",
        ".idea",
        ".vscode",
        ".dart_tool",
        "coverage",
        "htmlcov",
        ".eggs",
    }
)

# Extension -> language name. Files outside this set are skipped entirely
# for now; new languages get added here as parsers are built for them.
DEFAULT_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".dart": "dart",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
}

# Skip files larger than this — huge generated/vendored files add noise
# and slow embedding without adding useful search results.
DEFAULT_MAX_FILE_SIZE = 1_000_000  # 1 MB


def walk_files(
    root: Path,
    *,
    extensions: dict[str, str] | None = None,
    ignore_dirs: Iterable[str] | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> Iterator[Path]:
    """Yield every source file under `root` worth indexing.

    Directories in `ignore_dirs` (default: `DEFAULT_IGNORE_DIRS`) are
    pruned entirely rather than just filtered, so this stays fast even
    on repos with huge `node_modules`/`.venv` trees. Symlinks are not
    followed, to avoid infinite loops on self-referential links.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    exts = extensions if extensions is not None else DEFAULT_EXTENSIONS
    ignored = frozenset(ignore_dirs) if ignore_dirs is not None else DEFAULT_IGNORE_DIRS

    yield from _walk(root, exts, ignored, max_file_size)


def _walk(
    directory: Path,
    exts: dict[str, str],
    ignored: frozenset[str],
    max_file_size: int,
) -> Iterator[Path]:
    try:
        entries = sorted(directory.iterdir())
    except (PermissionError, OSError):
        return

    for entry in entries:
        if entry.is_symlink():
            continue

        if entry.is_dir():
            if entry.name in ignored or entry.name.startswith("."):
                continue
            yield from _walk(entry, exts, ignored, max_file_size)
        elif entry.is_file():
            if entry.suffix not in exts:
                continue
            try:
                if entry.stat().st_size > max_file_size:
                    continue
            except OSError:
                continue
            yield entry


def language_for(path: Path, extensions: dict[str, str] | None = None) -> str | None:
    """Return the language name registered for `path`'s extension, if any."""
    exts = extensions if extensions is not None else DEFAULT_EXTENSIONS
    return exts.get(path.suffix)
