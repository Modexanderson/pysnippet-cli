"""Regex + brace-matching snippet extraction for C-family languages
(JavaScript, TypeScript, Dart) that don't have a Python stdlib parser
available.

This is intentionally a lightweight heuristic, not a real parser: it
looks for common declaration shapes (function/class/arrow-function/
method) anchored at the start of a line, matches braces while skipping
strings and comments (see `brace_scanner`), and filters out
control-flow keywords that would otherwise look like function names
(`if (...) {`, `for (...) {`). Uncommon syntax -- decorators above a
class, generics with default values containing braces, expression-
bodied Dart functions (`int square(x) => x * x;`) -- may be missed.
Anything missed falls through to the generic splitter, so no content
is lost, just chunked less precisely. A tree-sitter-based rewrite
would close these gaps if they turn out to matter in practice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pysnippet_cli.parsers.brace_scanner import find_matching_brace
from pysnippet_cli.snippet import Snippet
from pysnippet_cli.splitter import split_generic

_CONTROL_KEYWORDS = frozenset(
    {"if", "for", "while", "switch", "catch", "do", "else", "try", "with", "function"}
)

_IDENT = r"[A-Za-z_$][\w$]*"

_JS_CLASS_RE = re.compile(
    rf"^[ \t]*(?:export\s+)?(?:default\s+)?class\s+(?P<name>{_IDENT})[^{{]*\{{",
    re.MULTILINE,
)
_JS_FUNCTION_RE = re.compile(
    rf"^[ \t]*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*"
    rf"(?P<name>{_IDENT})?\s*\([^)]*\)\s*(?::[^{{]+)?\{{",
    re.MULTILINE,
)
_JS_ARROW_RE = re.compile(
    rf"^[ \t]*(?:export\s+)?(?:const|let|var)\s+(?P<name>{_IDENT})\s*"
    rf"(?::[^=]+)?=\s*(?:async\s*)?\([^)]*\)\s*(?:=>|:[^=]+=>)\s*\{{",
    re.MULTILINE,
)
_JS_METHOD_RE = re.compile(
    rf"^[ \t]*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|readonly\s+|"
    rf"override\s+|get\s+|set\s+)*(?P<name>{_IDENT})\s*\([^)]*\)\s*(?::[^{{]+)?\{{",
    re.MULTILINE,
)

_DART_CLASS_RE = re.compile(
    rf"^[ \t]*(?:abstract\s+)?class\s+(?P<name>{_IDENT})[^{{;]*\{{",
    re.MULTILINE,
)
_DART_FUNCTION_RE = re.compile(
    rf"^[ \t]*(?:static\s+)?[\w$<>,\[\]?\s]+?\s+(?P<name>{_IDENT})\s*"
    rf"\([^;{{]*\)\s*(?:async\s*\*?|sync\s*\*?)?\s*\{{",
    re.MULTILINE,
)


@dataclass(frozen=True)
class _LanguageConfig:
    class_re: re.Pattern[str]
    function_re: re.Pattern[str]
    method_re: re.Pattern[str]
    arrow_re: re.Pattern[str] | None = None


_LANGUAGES: dict[str, _LanguageConfig] = {
    "javascript": _LanguageConfig(_JS_CLASS_RE, _JS_FUNCTION_RE, _JS_METHOD_RE, _JS_ARROW_RE),
    "typescript": _LanguageConfig(_JS_CLASS_RE, _JS_FUNCTION_RE, _JS_METHOD_RE, _JS_ARROW_RE),
    "dart": _LanguageConfig(_DART_CLASS_RE, _DART_FUNCTION_RE, _DART_FUNCTION_RE),
}


def supports(language: str) -> bool:
    return language in _LANGUAGES


def parse_brace_language(content: str, *, file_path: str, language: str) -> list[Snippet] | None:
    config = _LANGUAGES.get(language)
    if config is None:
        return None

    covered: list[tuple[int, int]] = []
    snippets: list[Snippet] = []

    cursor = 0
    length = len(content)
    while cursor < length:
        match, kind = _next_declaration(content, cursor, config)
        if match is None:
            break

        name = match.group("name")
        if name is None or name in _CONTROL_KEYWORDS:
            cursor = match.end()
            continue

        brace_index = match.end() - 1
        close_index = find_matching_brace(content, brace_index)
        if close_index is None:
            cursor = match.end()
            continue

        snippets.append(
            _make_snippet(
                content,
                match.start(),
                close_index,
                file_path=file_path,
                language=language,
                kind=kind,
                name=name,
            )
        )
        covered.append((match.start(), close_index))

        if kind == "class":
            body_start = brace_index + 1
            snippets.extend(
                _extract_methods(
                    content,
                    body_start,
                    close_index,
                    config.method_re,
                    file_path=file_path,
                    language=language,
                    class_name=name,
                )
            )

        cursor = close_index + 1

    snippets.extend(_leftover_snippets(content, covered, file_path=file_path, language=language))
    snippets.sort(key=lambda s: s.start_line)
    return snippets


def _next_declaration(
    content: str, cursor: int, config: _LanguageConfig
) -> tuple[re.Match[str] | None, str]:
    """Return whichever of class/function/arrow matches earliest at or
    after `cursor`, along with its snippet kind."""
    candidates: list[tuple[re.Match[str], str]] = []

    class_match = config.class_re.search(content, cursor)
    if class_match:
        candidates.append((class_match, "class"))

    func_match = config.function_re.search(content, cursor)
    if func_match:
        candidates.append((func_match, "function"))

    if config.arrow_re is not None:
        arrow_match = config.arrow_re.search(content, cursor)
        if arrow_match:
            candidates.append((arrow_match, "function"))

    if not candidates:
        return None, ""

    match, kind = min(candidates, key=lambda pair: pair[0].start())
    return match, kind


def _extract_methods(
    content: str,
    body_start: int,
    body_end: int,
    method_re: re.Pattern[str],
    *,
    file_path: str,
    language: str,
    class_name: str,
) -> list[Snippet]:
    snippets: list[Snippet] = []
    cursor = body_start

    while cursor < body_end:
        match = method_re.search(content, cursor, body_end)
        if match is None:
            break

        name = match.group("name")
        if name is None or name in _CONTROL_KEYWORDS:
            cursor = match.end()
            continue

        brace_index = match.end() - 1
        close_index = find_matching_brace(content, brace_index)
        if close_index is None or close_index > body_end:
            cursor = match.end()
            continue

        snippets.append(
            _make_snippet(
                content,
                match.start(),
                close_index,
                file_path=file_path,
                language=language,
                kind="method",
                name=f"{class_name}.{name}",
            )
        )
        cursor = close_index + 1

    return snippets


def _make_snippet(
    content: str,
    start_offset: int,
    end_offset: int,
    *,
    file_path: str,
    language: str,
    kind: str,
    name: str | None,
) -> Snippet:
    start_line = content.count("\n", 0, start_offset) + 1
    end_line = content.count("\n", 0, end_offset) + 1
    text = content[start_offset : end_offset + 1]
    return Snippet(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=text,
        kind=kind,
        name=name,
        language=language,
    )


def _leftover_snippets(
    content: str,
    covered: list[tuple[int, int]],
    *,
    file_path: str,
    language: str,
) -> list[Snippet]:
    if not covered:
        return split_generic(content, file_path=file_path, language=language)

    chars = list(content)
    for start, end in covered:
        for i in range(start, min(end + 1, len(chars))):
            if chars[i] != "\n":
                chars[i] = " "
    blanked = "".join(chars)
    return split_generic(blanked, file_path=file_path, language=language)
