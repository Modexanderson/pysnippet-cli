from pysnippet_cli.parsers.brace_scanner import find_matching_brace


def _open_at(text: str) -> int:
    return text.index("{")


class TestFindMatchingBrace:
    def test_simple_pair(self) -> None:
        text = "{}"
        assert find_matching_brace(text, 0) == 1

    def test_nested_braces(self) -> None:
        text = "{ { } }"
        close = find_matching_brace(text, 0)
        assert close == len(text) - 1

    def test_multiple_nesting_levels(self) -> None:
        text = "{ a { b { c } d } e }"
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_ignores_brace_in_double_quoted_string(self) -> None:
        text = '{ "}" }'
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_ignores_brace_in_single_quoted_string(self) -> None:
        text = "{ '}' }"
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_ignores_brace_in_template_literal(self) -> None:
        text = "{ `}` }"
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_ignores_brace_in_line_comment(self) -> None:
        text = "{ // }\n}"
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_ignores_brace_in_block_comment(self) -> None:
        text = "{ /* } */ }"
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_handles_escaped_quote_in_string(self) -> None:
        text = r'{ "a \" }" }'
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_handles_triple_quoted_string(self) -> None:
        text = "{ '''} }''' }"
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_returns_none_when_unclosed(self) -> None:
        text = "{ a b c"
        assert find_matching_brace(text, 0) is None

    def test_starts_from_given_open_index(self) -> None:
        text = "xxx { a } { b }"
        first_open = text.index("{")
        second_open = text.index("{", first_open + 1)
        assert find_matching_brace(text, second_open) == len(text) - 1

    def test_raises_if_index_not_a_brace(self) -> None:
        try:
            find_matching_brace("abc", 0)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_realistic_function_body(self) -> None:
        text = 'function foo() {\n  return "}";\n}'
        open_idx = _open_at(text)
        assert find_matching_brace(text, open_idx) == len(text) - 1

    def test_string_with_newline_bails_at_newline(self) -> None:
        # An unterminated single-quoted string shouldn't swallow the
        # rest of the file looking for a closing quote.
        text = "{ 'unterminated\n} "
        close = find_matching_brace(text, 0)
        assert close == text.index("}")

    def test_handles_escaped_char_inside_triple_quoted_string(self) -> None:
        text = r"{ '''a\'b''' }"
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_unterminated_triple_quoted_string_returns_none(self) -> None:
        text = "{ '''unterminated"
        assert find_matching_brace(text, 0) is None

    def test_unterminated_string_with_no_newline_returns_none(self) -> None:
        # A single-quoted string that runs to the end of the text
        # without a newline or closing quote (distinct from the
        # newline-bail case above -- this exercises the final `return i`
        # fallback at the end of the scan loop).
        text = "{ 'unterminated"
        assert find_matching_brace(text, 0) is None

    def test_unterminated_block_comment_returns_none(self) -> None:
        text = "{ /* unterminated"
        assert find_matching_brace(text, 0) is None
