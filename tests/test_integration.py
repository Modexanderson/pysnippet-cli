"""End-to-end integration tests exercising the real walk -> parse ->
embed -> store -> search pipeline together, with only the actual
neural network call swapped for a deterministic keyword-based fake so
relevance ordering can be asserted exactly.

Every other test file mocks embeddings with random vectors, which is
enough to prove the plumbing runs without crashing but can't prove
search actually *works* -- a bug that silently returns snippets in the
wrong order, or mixes up which file a snippet came from, would pass
those tests undetected. These tests catch that class of bug by using
real walker/parser/store/search-index code throughout, with a fake
that maps text to vectors predictably instead of randomly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from pysnippet_cli.embedding import EmbeddingModel
from pysnippet_cli.indexer import build_index, update_index
from pysnippet_cli.search_index import SearchIndex
from pysnippet_cli.store import SnippetStore

_VOCAB = [
    "sort",
    "database",
    "query",
    "format",
    "string",
    "network",
    "request",
    "math",
    "calculate",
]


def _keyword_vector(text: str) -> list[float]:
    lower = text.lower()
    vec = [1.0 if word in lower else 0.0 for word in _VOCAB]
    if not any(vec):
        vec[-1] = 0.01  # avoid an all-zero vector, which can't be L2-normalized
    return vec


def _deterministic_embedder(model_name: str = "fake-deterministic") -> EmbeddingModel:
    """A real EmbeddingModel whose lazy model load is swapped for a
    keyword-matching fake -- deterministic, not random, so search
    relevance through the real pipeline is actually assertable."""
    embedder = EmbeddingModel(model_name=model_name)
    fake_model = MagicMock()
    fake_model.get_embedding_dimension.return_value = len(_VOCAB)

    def _encode(texts, **kwargs):
        return np.array([_keyword_vector(t) for t in texts], dtype=np.float32)

    fake_model.encode.side_effect = _encode
    embedder._model = fake_model  # bypass lazy loading entirely
    return embedder


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _search(store: SnippetStore, embedder: EmbeddingModel, query: str, top_k: int = 5):
    ids, vectors = store.all_embeddings()
    index = SearchIndex(ids, vectors)
    query_vector = embedder.embed_texts([query])[0]
    results = index.search(query_vector, top_k=top_k)
    return [(store.get_snippet(sid), score) for sid, score in results]


class TestFullPipelineRelevance:
    def test_query_ranks_matching_topic_first(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "sorting.py",
            'def bubble_sort(items):\n    """Sort a list using bubble sort."""\n'
            "    return sorted(items)\n",
        )
        _write(
            tmp_path / "db.py",
            'def run_query(sql):\n    """Execute a database query."""\n'
            "    return execute(sql)\n",
        )
        _write(
            tmp_path / "fmt.py",
            'def format_string(s):\n    """Format a string value."""\n' "    return s.strip()\n",
        )

        embedder = _deterministic_embedder()
        result = build_index(tmp_path, embedder=embedder)
        assert result.snippets_indexed >= 3

        with SnippetStore(result.db_path) as store:
            results = _search(store, embedder, "sort items in a list", top_k=1)

        assert results[0][0].name == "bubble_sort"

    def test_different_queries_rank_different_topics(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "sorting.py",
            'def bubble_sort(items):\n    """Sort a list."""\n    return items\n',
        )
        _write(
            tmp_path / "db.py",
            'def run_query(sql):\n    """Run a database query."""\n    return sql\n',
        )
        _write(
            tmp_path / "fmt.py",
            'def format_string(s):\n    """Format a string."""\n    return s\n',
        )

        embedder = _deterministic_embedder()
        result = build_index(tmp_path, embedder=embedder)

        with SnippetStore(result.db_path) as store:
            sort_results = _search(store, embedder, "sort", top_k=1)
            db_results = _search(store, embedder, "database query", top_k=1)
            fmt_results = _search(store, embedder, "format a string", top_k=1)

        assert sort_results[0][0].name == "bubble_sort"
        assert db_results[0][0].name == "run_query"
        assert fmt_results[0][0].name == "format_string"

    def test_multi_language_index_tags_correct_language(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "a.py",
            'def sort_list(items):\n    """Sort items."""\n    return sorted(items)\n',
        )
        _write(
            tmp_path / "b.js",
            "function sortArray(items) {\n  // Sort an array.\n  return items.sort();\n}\n",
        )
        _write(tmp_path / "c.dart", "void sortWidget() {\n  // Sort widget items.\n}\n")

        embedder = _deterministic_embedder()
        result = build_index(tmp_path, embedder=embedder)
        assert result.files_scanned == 3

        with SnippetStore(result.db_path) as store:
            languages = {s.language for s in store.all_snippets()}
            assert languages == {"python", "javascript", "dart"}


class TestIncrementalUpdateIntegration:
    def test_new_content_becomes_searchable_after_update(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", 'def foo():\n    """Existing function."""\n    return 1\n')
        embedder = _deterministic_embedder()
        build_result = build_index(tmp_path, embedder=embedder)

        _write(
            tmp_path / "net.py",
            'def send_network_request(url):\n    """Send a network request."""\n'
            "    return fetch(url)\n",
        )
        update_index(tmp_path, embedder=embedder, db_path=build_result.db_path)

        with SnippetStore(build_result.db_path) as store:
            results = _search(store, embedder, "network request", top_k=1)

        assert results[0][0].name == "send_network_request"

    def test_removed_content_is_not_searchable_after_update(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", 'def foo():\n    """Existing function."""\n    return 1\n')
        _write(
            tmp_path / "math_ops.py",
            'def calculate_math(x):\n    """Calculate a math result."""\n    return x * 2\n',
        )
        embedder = _deterministic_embedder()
        build_result = build_index(tmp_path, embedder=embedder)

        (tmp_path / "math_ops.py").unlink()
        update_index(tmp_path, embedder=embedder, db_path=build_result.db_path)

        with SnippetStore(build_result.db_path) as store:
            names = {s.name for s in store.all_snippets()}
            assert "calculate_math" not in names

    def test_changed_content_reranks_correctly(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.py"
        _write(file_path, 'def process(x):\n    """Format a string value."""\n    return x\n')
        embedder = _deterministic_embedder()
        build_result = build_index(tmp_path, embedder=embedder)

        with SnippetStore(build_result.db_path) as store:
            before = _search(store, embedder, "database query", top_k=1)
        assert before[0][0].name != "process" or "database" not in before[0][0].content.lower()

        # Rewrite the same function to be about something else entirely.
        _write(file_path, 'def process(x):\n    """Run a database query."""\n    return x\n')
        import os

        os.utime(file_path, (file_path.stat().st_mtime + 100,) * 2)
        update_index(tmp_path, embedder=embedder, db_path=build_result.db_path)

        with SnippetStore(build_result.db_path) as store:
            after = _search(store, embedder, "database query", top_k=1)

        assert after[0][0].name == "process"


class TestRoundTripContent:
    def test_retrieved_snippet_content_matches_source(self, tmp_path: Path) -> None:
        source = 'def foo(a, b):\n    """Add two numbers."""\n    return a + b\n'
        _write(tmp_path / "a.py", source)

        embedder = _deterministic_embedder()
        result = build_index(tmp_path, embedder=embedder)

        with SnippetStore(result.db_path) as store:
            results = _search(store, embedder, "add two numbers", top_k=1)
            snippet, _ = results[0]

        assert snippet.content in source
        assert snippet.name == "foo"
        assert snippet.file_path == "a.py"

    def test_class_and_method_both_retrievable(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "widget.py",
            "class Sorter:\n"
            '    """A class for sorting things."""\n\n'
            "    def sort(self, items):\n"
            '        """Sort the given items."""\n'
            "        return sorted(items)\n",
        )
        embedder = _deterministic_embedder()
        result = build_index(tmp_path, embedder=embedder)

        with SnippetStore(result.db_path) as store:
            kinds = {s.kind for s in store.all_snippets()}
            names = {s.name for s in store.all_snippets()}

        assert "class" in kinds
        assert "method" in kinds
        assert "Sorter" in names
        assert "Sorter.sort" in names
