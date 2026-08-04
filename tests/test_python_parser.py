from pysnippet_cli.parsers.python_parser import parse_python


class TestTopLevelFunctions:
    def test_extracts_simple_function(self) -> None:
        content = "def foo():\n    return 1\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert len(functions) == 1
        assert functions[0].name == "foo"
        assert functions[0].start_line == 1
        assert functions[0].end_line == 2

    def test_extracts_multiple_functions(self) -> None:
        content = "def foo():\n    pass\n\n\ndef bar():\n    pass\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert {f.name for f in functions} == {"foo", "bar"}

    def test_async_function(self) -> None:
        content = "async def fetch():\n    return await get()\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert len(functions) == 1
        assert functions[0].name == "fetch"

    def test_decorator_included_in_span(self) -> None:
        content = "@staticmethod\n@another_decorator\ndef foo():\n    pass\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert len(functions) == 1
        assert functions[0].start_line == 1  # includes first decorator
        assert "@staticmethod" in functions[0].content

    def test_language_is_python(self) -> None:
        content = "def foo():\n    pass\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        assert snippets[0].language == "python"

    def test_file_path_propagates(self) -> None:
        content = "def foo():\n    pass\n"
        snippets = parse_python(content, file_path="src/mod.py")
        assert snippets is not None
        assert snippets[0].file_path == "src/mod.py"

    def test_docstring_included(self) -> None:
        content = 'def foo():\n    """Does a thing."""\n    return 1\n'
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert "Does a thing" in functions[0].content

    def test_nested_function_not_split_out(self) -> None:
        content = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert len(functions) == 1
        assert functions[0].name == "outer"
        assert "def inner" in functions[0].content


class TestClasses:
    def test_extracts_class(self) -> None:
        content = "class Foo:\n    pass\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        classes = [s for s in snippets if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "Foo"

    def test_extracts_methods_with_qualified_names(self) -> None:
        content = (
            "class Foo:\n"
            "    def bar(self):\n"
            "        return 1\n\n"
            "    def baz(self):\n"
            "        return 2\n"
        )
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        methods = [s for s in snippets if s.kind == "method"]
        assert {m.name for m in methods} == {"Foo.bar", "Foo.baz"}

    def test_class_snippet_spans_whole_class(self) -> None:
        content = "class Foo:\n    def bar(self):\n        return 1\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        classes = [s for s in snippets if s.kind == "class"]
        assert classes[0].start_line == 1
        assert classes[0].end_line == 3

    def test_async_method(self) -> None:
        content = "class Foo:\n    async def bar(self):\n        return await x()\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        methods = [s for s in snippets if s.kind == "method"]
        assert methods[0].name == "Foo.bar"

    def test_class_with_no_methods(self) -> None:
        content = "class Foo:\n    x = 1\n    y = 2\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        classes = [s for s in snippets if s.kind == "class"]
        methods = [s for s in snippets if s.kind == "method"]
        assert len(classes) == 1
        assert methods == []

    def test_multiple_classes(self) -> None:
        content = "class Foo:\n    pass\n\n\nclass Bar:\n    pass\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        classes = [s for s in snippets if s.kind == "class"]
        assert {c.name for c in classes} == {"Foo", "Bar"}


class TestLeftoverContent:
    def test_imports_captured_as_leftover(self) -> None:
        content = "import os\nimport sys\n\n\ndef foo():\n    pass\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        leftover = [s for s in snippets if s.kind == "block"]
        assert len(leftover) == 1
        assert "import os" in leftover[0].content

    def test_module_level_constant_captured(self) -> None:
        content = "def foo():\n    pass\n\n\nMAX_SIZE = 100\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        leftover = [s for s in snippets if s.kind == "block"]
        assert any("MAX_SIZE" in s.content for s in leftover)

    def test_file_with_only_leftover_content(self) -> None:
        content = "x = 1\ny = 2\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        assert len(snippets) == 1
        assert snippets[0].kind == "block"

    def test_snippets_sorted_by_start_line(self) -> None:
        content = "x = 1\n\n\ndef foo():\n    pass\n\n\ny = 2\n"
        snippets = parse_python(content, file_path="a.py")
        assert snippets is not None
        starts = [s.start_line for s in snippets]
        assert starts == sorted(starts)


class TestInvalidInput:
    def test_syntax_error_returns_none(self) -> None:
        content = "def foo(:\n    pass\n"
        assert parse_python(content, file_path="a.py") is None

    def test_empty_content(self) -> None:
        assert parse_python("", file_path="a.py") == []

    def test_only_whitespace(self) -> None:
        result = parse_python("   \n\n  \n", file_path="a.py")
        assert result == []
