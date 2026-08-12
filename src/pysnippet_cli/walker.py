"""Recursive directory walker that yields source files worth indexing."""

from __future__ import annotations

import fnmatch
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
    ignore_patterns: Iterable[str] | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> Iterator[Path]:
    """Yield every source file under `root` worth indexing.

    Directories in `ignore_dirs` (default: `DEFAULT_IGNORE_DIRS`) are
    pruned entirely rather than just filtered, so this stays fast even
    on repos with huge `node_modules`/`.venv` trees. Symlinks are not
    followed, to avoid infinite loops on self-referential links.

    `ignore_patterns` (e.g. from a `.pysnippetrc`) adds extra filtering:
    a pattern ending in `/` names an additional directory to prune
    (merged with `ignore_dirs`); any other pattern is matched via glob
    (`fnmatch`) against both the file's path relative to `root` and its
    bare filename, so `"*.generated.py"` and `"tests/fixtures/*"` both
    work as expected.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    exts = extensions if extensions is not None else DEFAULT_EXTENSIONS
    ignored = frozenset(ignore_dirs) if ignore_dirs is not None else DEFAULT_IGNORE_DIRS

    extra_ignore_dirs, file_patterns = _split_ignore_patterns(ignore_patterns or ())
    ignored = ignored | extra_ignore_dirs

    yield from _walk(root, root, exts, ignored, file_patterns, max_file_size)


def _split_ignore_patterns(patterns: Iterable[str]) -> tuple[frozenset[str], list[str]]:
    """Split raw ignore patterns into directory names to prune (patterns
    ending in "/") and glob patterns to match against file paths."""
    dir_names = set()
    file_patterns = []
    for pattern in patterns:
        if pattern.endswith("/"):
            dir_names.add(pattern.rstrip("/"))
        else:
            file_patterns.append(pattern)
    return frozenset(dir_names), file_patterns


def _matches_any(path: str, name: str, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(name, pattern) for pattern in patterns
    )


def _walk(
    directory: Path,
    root: Path,
    exts: dict[str, str],
    ignored: frozenset[str],
    file_patterns: list[str],
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
            yield from _walk(entry, root, exts, ignored, file_patterns, max_file_size)
        elif entry.is_file():
            if entry.suffix not in exts:
                continue
            if file_patterns:
                rel_path = entry.relative_to(root).as_posix()
                if _matches_any(rel_path, entry.name, file_patterns):
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


def extensions_for_languages(languages: Iterable[str] | None) -> dict[str, str] | None:
    """Return the extension -> language subset of `DEFAULT_EXTENSIONS`
    matching `languages` (e.g. from a `.pysnippetrc`), or None -- meaning
    no restriction -- when `languages` is None."""
    if languages is None:
        return None
    allowed = set(languages)
    return {ext: lang for ext, lang in DEFAULT_EXTENSIONS.items() if lang in allowed}
