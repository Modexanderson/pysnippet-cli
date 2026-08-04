"""Python-specific snippet extraction using the `ast` module.

Extracts top-level functions and classes (plus one level of methods
inside each class) as their own snippets. Nested functions/classes
inside a function body are not split out separately -- they stay part
of their enclosing function's snippet. Anything not covered by a
function/class (imports, module-level constants, `if __name__ ==`
blocks) is picked up by the generic blank-line splitter so nothing
from the file gets dropped.
"""

from __future__ import annotations

import ast

from pysnippet_cli.snippet import Snippet
from pysnippet_cli.splitter import split_generic

_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def parse_python(content: str, *, file_path: str) -> list[Snippet] | None:
    """Extract function/class/method snippets from Python source.

    Returns None if `content` isn't parseable as Python, so the caller
    can fall back to the generic splitter.
    """
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return None

    lines = content.splitlines()
    if not lines:
        return []

    covered: list[tuple[int, int]] = []  # 0-indexed inclusive line ranges
    snippets: list[Snippet] = []

    for node in tree.body:
        if isinstance(node, _DEF_TYPES):
            snippets.append(_snippet_for_node(node, lines, file_path, kind="function"))
            covered.append(_node_range(node))
        elif isinstance(node, ast.ClassDef):
            snippets.append(
                _snippet_for_node(node, lines, file_path, kind="class", name=node.name)
            )
            covered.append(_node_range(node))
            snippets.extend(_extract_methods(node, lines, file_path))

    snippets.extend(_leftover_snippets(lines, covered, file_path))
    snippets.sort(key=lambda s: s.start_line)
    return snippets


def _extract_methods(class_node: ast.ClassDef, lines: list[str], file_path: str) -> list[Snippet]:
    methods: list[Snippet] = []
    for child in class_node.body:
        if isinstance(child, _DEF_TYPES):
            qualified_name = f"{class_node.name}.{child.name}"
            methods.append(
                _snippet_for_node(
                    child, lines, file_path, kind="method", name=qualified_name
                )
            )
    return methods


def _node_range(node: ast.AST) -> tuple[int, int]:
    """0-indexed inclusive (start, end) line range for `node`, extended
    to include any decorators (whose lineno points at `def`/`class`
    itself, not the decorator lines above it)."""
    start = node.lineno - 1
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        start = min(start, decorators[0].lineno - 1)
    end = (node.end_lineno or node.lineno) - 1
    return start, end


def _snippet_for_node(
    node: ast.AST,
    lines: list[str],
    file_path: str,
    *,
    kind: str,
    name: str | None = None,
) -> Snippet:
    start, end = _node_range(node)
    text = "\n".join(lines[start : end + 1])
    return Snippet(
        file_path=file_path,
        start_line=start + 1,
        end_line=end + 1,
        content=text,
        kind=kind,
        name=name if name is not None else getattr(node, "name", None),
        language="python",
    )


def _leftover_snippets(
    lines: list[str], covered: list[tuple[int, int]], file_path: str
) -> list[Snippet]:
    """Blank out lines already claimed by a function/class, then run the
    generic splitter over what's left (imports, constants, etc.)."""
    if not covered:
        return split_generic("\n".join(lines), file_path=file_path, language="python")

    covered_set: set[int] = set()
    for start, end in covered:
        covered_set.update(range(start, end + 1))

    leftover_lines = [line if i not in covered_set else "" for i, line in enumerate(lines)]
    leftover_content = "\n".join(leftover_lines)
    return split_generic(leftover_content, file_path=file_path, language="python")
