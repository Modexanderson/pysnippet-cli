import io

from rich.console import Console

from pysnippet_cli.display import lexer_for, render_snippet
from pysnippet_cli.snippet import Snippet


def _render(snippet: Snippet) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    render_snippet(snippet, console=console)
    return buffer.getvalue()


class TestLexerFor:
    def test_maps_generic_to_text(self) -> None:
        assert lexer_for("generic") == "text"

    def test_passes_through_known_languages(self) -> None:
        assert lexer_for("python") == "python"
        assert lexer_for("javascript") == "javascript"
        assert lexer_for("typescript") == "typescript"
        assert lexer_for("dart") == "dart"
        assert lexer_for("go") == "go"
        assert lexer_for("rust") == "rust"


class TestRenderSnippet:
    def test_includes_content(self) -> None:
        snippet = Snippet(
            file_path="a.py",
            start_line=1,
            end_line=2,
            content="def foo():\n    return 1",
            kind="function",
            name="foo",
            language="python",
        )
        output = _render(snippet)
        assert "def foo" in output
        assert "return 1" in output

    def test_includes_title_with_kind_and_name(self) -> None:
        snippet = Snippet(
            file_path="a.py",
            start_line=1,
            end_line=1,
            content="x = 1",
            kind="function",
            name="foo",
        )
        output = _render(snippet)
        assert "function foo" in output

    def test_title_omits_name_when_none(self) -> None:
        snippet = Snippet(
            file_path="a.py", start_line=1, end_line=1, content="x = 1", kind="block", name=None
        )
        output = _render(snippet)
        assert "block" in output

    def test_includes_location_and_id(self) -> None:
        snippet = Snippet(
            file_path="src/foo.py", start_line=10, end_line=15, content="x = 1", name="foo"
        )
        output = _render(snippet)
        assert "src/foo.py" in output
        assert "10" in output
        assert "15" in output
        assert snippet.id in output

    def test_line_numbers_start_at_snippet_start_line(self) -> None:
        snippet = Snippet(
            file_path="a.py",
            start_line=42,
            end_line=44,
            content="a = 1\nb = 2\nc = 3",
            language="python",
        )
        output = _render(snippet)
        assert "42" in output
        assert "44" in output

    def test_generic_language_does_not_raise(self) -> None:
        snippet = Snippet(
            file_path="a.txt", start_line=1, end_line=1, content="hello", language="generic"
        )
        # Should not raise even though "generic" isn't a real Pygments lexer name.
        output = _render(snippet)
        assert "hello" in output

    def test_unknown_language_falls_back_gracefully(self) -> None:
        snippet = Snippet(
            file_path="a.xyz", start_line=1, end_line=1, content="???", language="totally-fake"
        )
        # Pygments falls back to a plain-text-like lexer for unrecognized
        # names rather than raising -- verify we don't crash either.
        output = _render(snippet)
        assert "???" in output

    def test_creates_own_console_when_none_given(self) -> None:
        snippet = Snippet(file_path="a.py", start_line=1, end_line=1, content="x = 1")
        # Just verify it doesn't raise when no console is injected.
        render_snippet(snippet)
