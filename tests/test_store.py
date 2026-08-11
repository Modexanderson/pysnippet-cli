from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysnippet_cli.snippet import Snippet
from pysnippet_cli.store import SnippetStore


def _snippet(
    file_path: str = "a.py",
    start_line: int = 1,
    end_line: int = 2,
    content: str = "def foo(): pass",
    kind: str = "function",
    name: str | None = "foo",
    language: str = "python",
) -> Snippet:
    return Snippet(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        kind=kind,
        name=name,
        language=language,
    )


class TestSchemaAndLifecycle:
    def test_creates_db_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "index.db"
        with SnippetStore(db_path):
            pass
        assert db_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "dir" / "index.db"
        with SnippetStore(db_path):
            pass
        assert db_path.exists()

    def test_empty_store_count_is_zero(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            assert store.count() == 0

    def test_reopening_existing_db_preserves_data(self, tmp_path: Path) -> None:
        db_path = tmp_path / "index.db"
        with SnippetStore(db_path) as store:
            store.add_snippets([_snippet()], np.zeros((1, 4), dtype=np.float32))

        with SnippetStore(db_path) as store:
            assert store.count() == 1


class TestAddSnippets:
    def test_adds_and_retrieves(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            snippet = _snippet()
            embeddings = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
            store.add_snippets([snippet], embeddings)

            assert store.count() == 1
            retrieved = store.get_snippet(snippet.id)
            assert retrieved is not None
            assert retrieved.file_path == "a.py"
            assert retrieved.name == "foo"
            assert retrieved.content == "def foo(): pass"

    def test_empty_list_is_noop(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.add_snippets([], np.empty((0, 4), dtype=np.float32))
            assert store.count() == 0

    def test_raises_on_length_mismatch(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            snippets = [_snippet(), _snippet(file_path="b.py")]
            embeddings = np.zeros((1, 4), dtype=np.float32)
            with pytest.raises(ValueError):
                store.add_snippets(snippets, embeddings)

    def test_reindexing_same_id_overwrites(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            snippet = _snippet(content="def foo(): pass")
            store.add_snippets([snippet], np.zeros((1, 4), dtype=np.float32))

            # Same id (same file/lines/content) but different embedding
            new_embedding = np.ones((1, 4), dtype=np.float32)
            store.add_snippets([snippet], new_embedding)

            assert store.count() == 1
            ids, vectors = store.all_embeddings()
            assert np.array_equal(vectors[0], new_embedding[0])

    def test_multiple_snippets_preserve_all_fields(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            snippets = [
                _snippet(file_path="a.py", name="foo", kind="function"),
                _snippet(file_path="b.py", name="Bar.baz", kind="method", language="typescript"),
            ]
            embeddings = np.random.rand(2, 4).astype(np.float32)
            store.add_snippets(snippets, embeddings)

            retrieved = {s.id: s for s in store.all_snippets()}
            assert retrieved[snippets[0].id].kind == "function"
            assert retrieved[snippets[1].id].kind == "method"
            assert retrieved[snippets[1].id].language == "typescript"

    def test_name_can_be_none(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            snippet = _snippet(name=None, kind="block")
            store.add_snippets([snippet], np.zeros((1, 4), dtype=np.float32))

            retrieved = store.get_snippet(snippet.id)
            assert retrieved is not None
            assert retrieved.name is None


class TestGetSnippet:
    def test_returns_none_for_missing_id(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            assert store.get_snippet("does-not-exist") is None


class TestDeleteByFile:
    def test_removes_only_matching_file(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            a = _snippet(file_path="a.py")
            b = _snippet(file_path="b.py")
            store.add_snippets([a, b], np.zeros((2, 4), dtype=np.float32))

            store.delete_by_file("a.py")

            assert store.count() == 1
            assert store.get_snippet(a.id) is None
            assert store.get_snippet(b.id) is not None


class TestClear:
    def test_removes_all_snippets(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.add_snippets([_snippet()], np.zeros((1, 4), dtype=np.float32))
            store.clear()
            assert store.count() == 0

    def test_removes_meta(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.set_meta("model", "all-MiniLM-L6-v2")
            store.clear()
            assert store.get_meta("model") is None


class TestMeta:
    def test_round_trip(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.set_meta("model_name", "all-MiniLM-L6-v2")
            assert store.get_meta("model_name") == "all-MiniLM-L6-v2"

    def test_missing_key_returns_none(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            assert store.get_meta("nonexistent") is None

    def test_overwrite_existing_key(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.set_meta("model_name", "model-v1")
            store.set_meta("model_name", "model-v2")
            assert store.get_meta("model_name") == "model-v2"


class TestAllEmbeddings:
    def test_empty_store(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            ids, vectors = store.all_embeddings()
            assert ids == []
            assert vectors.shape[0] == 0

    def test_returns_all_ids_and_vectors(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            snippets = [_snippet(file_path="a.py"), _snippet(file_path="b.py")]
            embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
            store.add_snippets(snippets, embeddings)

            ids, vectors = store.all_embeddings()
            assert set(ids) == {snippets[0].id, snippets[1].id}
            assert vectors.shape == (2, 4)

    def test_vectors_match_inserted_values_exactly(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            snippet = _snippet()
            embedding = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
            store.add_snippets([snippet], embedding)

            ids, vectors = store.all_embeddings()
            assert np.array_equal(vectors[0], embedding[0])


class TestAllSnippets:
    def test_ordered_by_file_path_then_line(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            snippets = [
                _snippet(file_path="b.py", start_line=1),
                _snippet(file_path="a.py", start_line=5, content="x"),
                _snippet(file_path="a.py", start_line=1, content="y"),
            ]
            store.add_snippets(snippets, np.zeros((3, 4), dtype=np.float32))

            result = store.all_snippets()
            paths_and_lines = [(s.file_path, s.start_line) for s in result]
            assert paths_and_lines == [("a.py", 1), ("a.py", 5), ("b.py", 1)]

    def test_empty_store_returns_empty_list(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            assert store.all_snippets() == []


class TestFileRecords:
    def test_round_trip(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.set_file_record("a.py", 123.456, "abc123")
            record = store.get_file_record("a.py")
            assert record == (123.456, "abc123")

    def test_missing_record_returns_none(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            assert store.get_file_record("nonexistent.py") is None

    def test_overwrite_existing_record(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.set_file_record("a.py", 100.0, "hash1")
            store.set_file_record("a.py", 200.0, "hash2")
            assert store.get_file_record("a.py") == (200.0, "hash2")

    def test_delete_file_record(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.set_file_record("a.py", 100.0, "hash1")
            store.delete_file_record("a.py")
            assert store.get_file_record("a.py") is None

    def test_delete_nonexistent_record_is_noop(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.delete_file_record("nonexistent.py")  # should not raise

    def test_all_file_paths(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.set_file_record("a.py", 1.0, "h1")
            store.set_file_record("b.py", 2.0, "h2")
            assert store.all_file_paths() == {"a.py", "b.py"}

    def test_all_file_paths_empty(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            assert store.all_file_paths() == set()

    def test_clear_removes_file_records(self, tmp_path: Path) -> None:
        with SnippetStore(tmp_path / "index.db") as store:
            store.set_file_record("a.py", 1.0, "h1")
            store.clear()
            assert store.all_file_paths() == set()
