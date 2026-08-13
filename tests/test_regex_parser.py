from pysnippet_cli.parsers.regex_parser import (
    _JS_METHOD_RE,
    _extract_methods,
    parse_brace_language,
    supports,
)


class TestSupports:
    def test_known_languages(self) -> None:
        assert supports("javascript")
        assert supports("typescript")
        assert supports("dart")

    def test_unknown_language(self) -> None:
        assert not supports("python")
        assert not supports("go")


class TestJavaScriptFunctions:
    def test_function_declaration(self) -> None:
        content = "function foo() {\n  return 1;\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert len(functions) == 1
        assert functions[0].name == "foo"
        assert functions[0].start_line == 1
        assert functions[0].end_line == 3

    def test_async_function(self) -> None:
        content = "async function fetchData() {\n  return await get();\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert functions[0].name == "fetchData"

    def test_export_function(self) -> None:
        content = "export function foo() {\n  return 1;\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert functions[0].name == "foo"

    def test_export_default_function(self) -> None:
        content = "export default function foo() {\n  return 1;\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert functions[0].name == "foo"

    def test_multiple_functions(self) -> None:
        content = "function foo() {\n  return 1;\n}\n\nfunction bar() {\n  return 2;\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert {f.name for f in functions} == {"foo", "bar"}

    def test_arrow_function_const(self) -> None:
        content = "const foo = () => {\n  return 1;\n};\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert functions[0].name == "foo"

    def test_arrow_function_with_params(self) -> None:
        content = "const add = (a, b) => {\n  return a + b;\n};\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert functions[0].name == "add"

    def test_async_arrow_function(self) -> None:
        content = "const load = async () => {\n  return await fetch();\n};\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert functions[0].name == "load"

    def test_does_not_false_positive_on_if_statement(self) -> None:
        content = "function foo() {\n  if (x) {\n    return 1;\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert len(functions) == 1
        assert functions[0].name == "foo"

    def test_brace_inside_string_does_not_break_extraction(self) -> None:
        content = 'function foo() {\n  return "}";\n}\n\nfunction bar() {\n  return 2;\n}\n'
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert {f.name for f in functions} == {"foo", "bar"}


class TestJavaScriptClasses:
    def test_class_declaration(self) -> None:
        content = "class Foo {\n  bar() {\n    return 1;\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        classes = [s for s in snippets if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "Foo"

    def test_class_with_extends(self) -> None:
        content = "class Foo extends Bar {\n  baz() {\n    return 1;\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        classes = [s for s in snippets if s.kind == "class"]
        assert classes[0].name == "Foo"

    def test_methods_qualified_with_class_name(self) -> None:
        content = "class Foo {\n  bar() {\n    return 1;\n  }\n\n  baz() {\n    return 2;\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        methods = [s for s in snippets if s.kind == "method"]
        assert {m.name for m in methods} == {"Foo.bar", "Foo.baz"}

    def test_async_method(self) -> None:
        content = "class Foo {\n  async bar() {\n    return await x();\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        methods = [s for s in snippets if s.kind == "method"]
        assert methods[0].name == "Foo.bar"

    def test_static_method(self) -> None:
        content = "class Foo {\n  static bar() {\n    return 1;\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        methods = [s for s in snippets if s.kind == "method"]
        assert methods[0].name == "Foo.bar"

    def test_export_class(self) -> None:
        content = "export class Foo {\n  bar() {\n    return 1;\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        classes = [s for s in snippets if s.kind == "class"]
        assert classes[0].name == "Foo"

    def test_method_body_not_scanned_for_nested_methods(self) -> None:
        content = (
            "class Foo {\n"
            "  bar() {\n"
            "    const helper = () => {\n"
            "      return 1;\n"
            "    };\n"
            "    return helper();\n"
            "  }\n"
            "}\n"
        )
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        methods = [s for s in snippets if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "Foo.bar"


class TestTypeScript:
    def test_typed_function(self) -> None:
        content = "function add(a: number, b: number): number {\n  return a + b;\n}\n"
        snippets = parse_brace_language(content, file_path="a.ts", language="typescript")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert functions[0].name == "add"

    def test_interface_implementing_class(self) -> None:
        content = "class Foo implements Bar {\n  baz(): void {\n    return;\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.ts", language="typescript")
        assert snippets is not None
        classes = [s for s in snippets if s.kind == "class"]
        assert classes[0].name == "Foo"


class TestDart:
    def test_simple_function(self) -> None:
        content = "void main() {\n  print('hello');\n}\n"
        snippets = parse_brace_language(content, file_path="a.dart", language="dart")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert functions[0].name == "main"

    def test_class_declaration(self) -> None:
        content = "class Foo {\n  void bar() {\n    return;\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.dart", language="dart")
        assert snippets is not None
        classes = [s for s in snippets if s.kind == "class"]
        assert classes[0].name == "Foo"

    def test_class_with_extends(self) -> None:
        content = "class Foo extends StatelessWidget {\n  void build() {\n    return;\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.dart", language="dart")
        assert snippets is not None
        classes = [s for s in snippets if s.kind == "class"]
        assert classes[0].name == "Foo"

    def test_generic_return_type(self) -> None:
        content = "Future<void> loadData() async {\n  await fetch();\n}\n"
        snippets = parse_brace_language(content, file_path="a.dart", language="dart")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert functions[0].name == "loadData"

    def test_does_not_false_positive_on_for_loop(self) -> None:
        content = "void main() {\n  for (var i = 0; i < 10; i++) {\n    print(i);\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.dart", language="dart")
        assert snippets is not None
        functions = [s for s in snippets if s.kind == "function"]
        assert len(functions) == 1
        assert functions[0].name == "main"


class TestLeftoverAndOrdering:
    def test_imports_captured_as_leftover(self) -> None:
        content = "import 'foo.dart';\n\nvoid main() {\n  print('hi');\n}\n"
        snippets = parse_brace_language(content, file_path="a.dart", language="dart")
        assert snippets is not None
        leftover = [s for s in snippets if s.kind == "block"]
        assert any("import" in s.content for s in leftover)

    def test_snippets_sorted_by_start_line(self) -> None:
        content = "const x = 1;\n\nfunction foo() {\n  return 1;\n}\n\nconst y = 2;\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        starts = [s.start_line for s in snippets]
        assert starts == sorted(starts)

    def test_unsupported_language_returns_none(self) -> None:
        assert parse_brace_language("x = 1", file_path="a.go", language="go") is None

    def test_empty_content(self) -> None:
        result = parse_brace_language("", file_path="a.js", language="javascript")
        assert result == []


class TestDefensiveEdgeCases:
    def test_anonymous_default_export_function_is_skipped(self) -> None:
        # JS_FUNCTION_RE's name group is optional -- an anonymous
        # function has no name to index it by, so it's left as leftover
        # content rather than becoming a nameless "function" snippet.
        content = "export default function() {\n  return 1;\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        assert all(s.kind != "function" for s in snippets)

    def test_unclosed_top_level_declaration_is_skipped(self) -> None:
        # A function whose brace never closes anywhere in the file --
        # find_matching_brace returns None, so it's never extracted.
        content = "function foo() {\n  return 1;\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        assert all(s.kind != "function" for s in snippets)

    def test_control_keyword_inside_class_body_is_not_a_method(self) -> None:
        # The loose method pattern doesn't require a keyword, so it can
        # match control-flow-shaped text like "if (x) {" -- the control
        # keyword filter catches this rather than emitting a bogus
        # "Foo.if" method.
        content = "class Foo {\n  if (x) {\n    return 1;\n  }\n}\n"
        snippets = parse_brace_language(content, file_path="a.js", language="javascript")
        assert snippets is not None
        assert all(s.name != "Foo.if" for s in snippets)

    def test_extract_methods_skips_unclosed_method_brace(self) -> None:
        content = "  bar() {\n    return 1;\n"
        result = _extract_methods(
            content,
            0,
            len(content),
            _JS_METHOD_RE,
            file_path="a.js",
            language="javascript",
            class_name="Foo",
        )
        assert result == []
