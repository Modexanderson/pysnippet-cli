"""Rich-formatted snippet display with syntax highlighting."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from pysnippet_cli.snippet import Snippet

# Our internal language names line up with Pygments lexer aliases for
# every language the walker/parsers recognize, except "generic" (files
# with no dedicated parser) which Pygments has no lexer named that --
# map it to plain text instead of letting Syntax raise on lookup.
_LEXER_OVERRIDES = {"generic": "text"}


def lexer_for(language: str) -> str:
    return _LEXER_OVERRIDES.get(language, language)


def render_snippet(snippet: Snippet, *, console: Console | None = None) -> None:
    """Pretty-print a snippet with syntax highlighting, line numbers
    matching its position in the source file, and a header showing its
    kind/name/location/id."""
    console = console or Console()

    title = snippet.kind if snippet.name is None else f"{snippet.kind} {snippet.name}"
    subtitle = f"{snippet.file_path}:{snippet.start_line}-{snippet.end_line}  id={snippet.id}"

    syntax = Syntax(
        snippet.content,
        lexer_for(snippet.language),
        theme="monokai",
        line_numbers=True,
        start_line=snippet.start_line,
        word_wrap=False,
    )

    console.print(Panel(syntax, title=title, subtitle=subtitle, border_style="cyan"))
