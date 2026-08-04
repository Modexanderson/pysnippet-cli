"""Generic, language-agnostic snippet splitting.

Splits a file into blocks separated by blank lines, merges blocks that
are too small to be useful on their own, and caps oversized blocks at
a maximum line count. This is the fallback splitter used for any file
that doesn't have a dedicated language-aware parser (see Day 3+ for
Python AST / JS-TS / Dart parsers, which will take priority over this
one when available).
"""

from __future__ import annotations

from pathlib import Path

from pysnippet_cli.snippet import Snippet

DEFAULT_MAX_LINES = 60
DEFAULT_MIN_LINES = 3

# Encodings tried in order when reading a source file. utf-8-sig
# strips a BOM if present and decodes plain UTF-8 identically to
# "utf-8" otherwise, so it covers both cases in one pass. latin-1
# never raises (every byte is a valid code point), so it's the last
# resort for legacy-encoded files rather than skipping them entirely.
_ENCODINGS = ("utf-8-sig", "latin-1")


def read_text(path: Path) -> str | None:
    """Read `path` as text, returning None if it can't be decoded or is binary."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None

    if b"\x00" in raw:
        return None  # binary file

    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def split_generic(
    content: str,
    *,
    file_path: str,
    language: str = "generic",
    max_lines: int = DEFAULT_MAX_LINES,
    min_lines: int = DEFAULT_MIN_LINES,
) -> list[Snippet]:
    """Split `content` into blank-line-delimited blocks.

    Blocks smaller than `min_lines` are merged into an adjacent block
    so trivial one-liners don't become their own low-signal snippets.
    Blocks larger than `max_lines` are cut into consecutive chunks of
    `max_lines` so no single snippet dominates embedding/search cost.
    """
    lines = content.splitlines()
    if not lines:
        return []

    blocks = _group_into_blocks(lines)
    blocks = _merge_small_blocks(blocks, min_lines=min_lines)

    snippets: list[Snippet] = []
    for start, end in blocks:
        snippets.extend(
            _chunk_block(
                lines, start, end, file_path=file_path, language=language, max_lines=max_lines
            )
        )
    return snippets


def _group_into_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Group 0-indexed line numbers into (start, end) inclusive ranges,
    splitting wherever a blank line occurs."""
    blocks: list[tuple[int, int]] = []
    start: int | None = None

    for i, line in enumerate(lines):
        if line.strip():
            if start is None:
                start = i
        else:
            if start is not None:
                blocks.append((start, i - 1))
                start = None

    if start is not None:
        blocks.append((start, len(lines) - 1))

    return blocks


def _merge_small_blocks(
    blocks: list[tuple[int, int]], *, min_lines: int
) -> list[tuple[int, int]]:
    """Merge blocks with fewer than `min_lines` lines into the next block
    (or the previous one, if it's the last block in the file)."""
    if not blocks:
        return blocks

    merged: list[tuple[int, int]] = []
    pending_start: int | None = None

    for start, end in blocks:
        size = end - start + 1
        block_start = pending_start if pending_start is not None else start

        if size < min_lines:
            pending_start = block_start
            continue

        merged.append((block_start, end))
        pending_start = None

    if pending_start is not None:
        # Trailing small block(s) with nothing after them to merge into —
        # attach to the previous merged block if one exists, else keep as-is.
        if merged:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, blocks[-1][1])
        else:
            merged.append((pending_start, blocks[-1][1]))

    return merged


def _chunk_block(
    lines: list[str],
    start: int,
    end: int,
    *,
    file_path: str,
    language: str,
    max_lines: int,
) -> list[Snippet]:
    """Split a single (start, end) 0-indexed range into snippets of at
    most `max_lines` lines each."""
    snippets: list[Snippet] = []
    pos = start

    while pos <= end:
        chunk_end = min(pos + max_lines - 1, end)
        text = "\n".join(lines[pos : chunk_end + 1])
        snippets.append(
            Snippet(
                file_path=file_path,
                start_line=pos + 1,
                end_line=chunk_end + 1,
                content=text,
                kind="block",
                language=language,
            )
        )
        pos = chunk_end + 1

    return snippets
