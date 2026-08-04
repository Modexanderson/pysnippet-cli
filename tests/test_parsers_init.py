from pysnippet_cli.parsers import parse


class TestDispatch:
    def test_python_uses_ast_parser(self) -> None:
        content = "def foo():\n    pass\n"
        snippets = parse(content, file_path="a.py", language="python")
        assert any(s.kind == "function" and s.name == "foo" for s in snippets)

    def test_python_syntax_error_falls_back_to_generic(self) -> None:
        content = "def foo(:\n    pass\n"
        snippets = parse(content, file_path="a.py", language="python")
        assert len(snippets) == 1
        assert snippets[0].kind == "block"

    def test_javascript_uses_regex_parser(self) -> None:
        content = "function foo() {\n  return 1;\n}\n"
        snippets = parse(content, file_path="a.js", language="javascript")
        assert any(s.kind == "function" and s.name == "foo" for s in snippets)

    def test_dart_uses_regex_parser(self) -> None:
        content = "void main() {\n  print(1);\n}\n"
        snippets = parse(content, file_path="a.dart", language="dart")
        assert any(s.kind == "function" and s.name == "main" for s in snippets)

    def test_unsupported_language_falls_back_to_generic(self) -> None:
        content = "func foo() {\n\treturn 1\n}\n"
        snippets = parse(content, file_path="a.go", language="go")
        assert len(snippets) == 1
        assert snippets[0].kind == "block"
        assert snippets[0].language == "go"

    def test_empty_content_returns_empty(self) -> None:
        assert parse("", file_path="a.py", language="python") == []
        assert parse("", file_path="a.js", language="javascript") == []
        assert parse("", file_path="a.go", language="go") == []
