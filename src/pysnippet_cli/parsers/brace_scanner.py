"""Low-level scanning helpers for brace-delimited, C-family languages
(JavaScript, TypeScript, Dart, and similar).

Used to find the matching closing brace for a declaration while
skipping over braces that appear inside string and comment literals,
so a `{` inside a string like `"{"` doesn't throw off the count.
"""

from __future__ import annotations


def find_matching_brace(text: str, open_index: int) -> int | None:
    """Given the index of an opening '{' in `text`, return the index of
    its matching closing '}', skipping braces inside strings/comments.

    Returns None if no match is found before the end of `text`.
    """
    if text[open_index] != "{":
        raise ValueError("open_index must point at '{'")

    depth = 0
    i = open_index
    length = len(text)

    while i < length:
        ch = text[i]

        if ch == "{":
            depth += 1
            i += 1
        elif ch == "}":
            depth -= 1
            i += 1
            if depth == 0:
                return i - 1
        elif ch in ("'", '"', "`"):
            i = _skip_string(text, i)
        elif ch == "/" and i + 1 < length and text[i + 1] == "/":
            i = _skip_line_comment(text, i)
        elif ch == "/" and i + 1 < length and text[i + 1] == "*":
            i = _skip_block_comment(text, i)
        else:
            i += 1

    return None


def _skip_string(text: str, i: int) -> int:
    """Advance past a string literal starting at `i` (which must point
    at a quote character). Handles single/double/backtick quotes, their
    triple-quoted variants (used by Dart), and backslash escapes."""
    quote = text[i]
    length = len(text)
    triple = text[i : i + 3] == quote * 3

    if triple:
        i += 3
        while i < length:
            if text[i] == "\\":
                i += 2
                continue
            if text[i : i + 3] == quote * 3:
                return i + 3
            i += 1
        return length

    i += 1
    while i < length:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i + 1
        if text[i] == "\n" and quote != "`":
            # Single/double-quoted strings can't legally span lines --
            # bail at the newline instead of scanning past it, in case
            # this was actually a false-positive match (e.g. an
            # apostrophe inside a comment we failed to detect).
            return i
        i += 1
    return i


def _skip_line_comment(text: str, i: int) -> int:
    length = len(text)
    while i < length and text[i] != "\n":
        i += 1
    return i


def _skip_block_comment(text: str, i: int) -> int:
    length = len(text)
    i += 2
    while i + 1 < length:
        if text[i] == "*" and text[i + 1] == "/":
            return i + 2
        i += 1
    return length
