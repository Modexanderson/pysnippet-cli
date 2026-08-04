"""Language-aware snippet parsers.

Dispatches to a dedicated parser for languages that have one (Python
via `ast`, JS/TS/Dart via regex + brace matching), falling back to the
generic blank-line splitter for anything else.
"""

from __future__ import annotations

from pysnippet_cli.parsers.python_parser import parse_python
from pysnippet_cli.parsers.regex_parser import parse_brace_language
from pysnippet_cli.snippet import Snippet
from pysnippet_cli.splitter import split_generic


def parse(content: str, *, file_path: str, language: str) -> list[Snippet]:
    """Extract snippets from `content` using the best available parser
    for `language`, falling back to generic block splitting.

    Never raises on malformed input -- worst case, everything ends up
    in generic blocks.
    """
    if language == "python":
        result = parse_python(content, file_path=file_path)
        if result is not None:
            return result
        return split_generic(content, file_path=file_path, language=language)

    result = parse_brace_language(content, file_path=file_path, language=language)
    if result is not None:
        return result

    return split_generic(content, file_path=file_path, language=language)
