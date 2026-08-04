from pysnippet_cli.snippet import Snippet


def test_id_is_deterministic() -> None:
    a = Snippet(file_path="foo.py", start_line=1, end_line=3, content="def foo(): pass")
    b = Snippet(file_path="foo.py", start_line=1, end_line=3, content="def foo(): pass")
    assert a.id == b.id


def test_id_differs_for_different_content() -> None:
    a = Snippet(file_path="foo.py", start_line=1, end_line=3, content="def foo(): pass")
    b = Snippet(file_path="foo.py", start_line=1, end_line=3, content="def bar(): pass")
    assert a.id != b.id


def test_id_differs_for_different_location() -> None:
    a = Snippet(file_path="foo.py", start_line=1, end_line=3, content="x = 1")
    b = Snippet(file_path="bar.py", start_line=1, end_line=3, content="x = 1")
    assert a.id != b.id


def test_explicit_id_is_respected() -> None:
    s = Snippet(file_path="foo.py", start_line=1, end_line=1, content="x", id="custom")
    assert s.id == "custom"


def test_line_count() -> None:
    s = Snippet(file_path="foo.py", start_line=5, end_line=9, content="...")
    assert s.line_count == 5


def test_line_count_single_line() -> None:
    s = Snippet(file_path="foo.py", start_line=5, end_line=5, content="x = 1")
    assert s.line_count == 1


def test_location() -> None:
    s = Snippet(file_path="src/foo.py", start_line=42, end_line=50, content="...")
    assert s.location() == "src/foo.py:42"


def test_defaults() -> None:
    s = Snippet(file_path="foo.py", start_line=1, end_line=1, content="x")
    assert s.kind == "block"
    assert s.name is None
    assert s.language == "generic"


def test_is_frozen() -> None:
    s = Snippet(file_path="foo.py", start_line=1, end_line=1, content="x")
    try:
        s.content = "y"  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except AttributeError:
        pass
