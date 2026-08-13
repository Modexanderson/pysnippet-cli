import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pysnippet_cli.embedding import EmbeddingModel
from pysnippet_cli.indexer import (
    build_index,
    find_project_index,
    index_path_for,
    update_index,
)
from pysnippet_cli.store import SnippetStore


def _fake_embedder(dim: int = 4) -> EmbeddingModel:
    """A real EmbeddingModel with its model-loading swapped for a fake
    that returns deterministic vectors, so tests don't download or run
    the actual transformer model."""
    embedder = EmbeddingModel(model_name="fake-model")
    fake_model = MagicMock()
    fake_model.get_embedding_dimension.return_value = dim

    def _encode(texts, **kwargs):
        return np.random.rand(len(texts), dim).astype(np.float32)

    fake_model.encode.side_effect = _encode
    embedder._model = fake_model  # bypass lazy loading entirely
    return embedder


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestIndexPathFor:
    def test_default_location(self, tmp_path: Path) -> None:
        result = index_path_for(tmp_path)
        assert result == tmp_path / ".pysnippet" / "index.db"


class TestFindProjectIndex:
    def test_finds_index_in_start_directory(self, tmp_path: Path) -> None:
        index_dir = tmp_path / ".pysnippet"
        index_dir.mkdir()
        (index_dir / "index.db").write_text("", encoding="utf-8")

        result = find_project_index(tmp_path)
        assert result == tmp_path / ".pysnippet" / "index.db"

    def test_finds_index_in_parent_directory(self, tmp_path: Path) -> None:
        index_dir = tmp_path / ".pysnippet"
        index_dir.mkdir()
        (index_dir / "index.db").write_text("", encoding="utf-8")

        nested = tmp_path / "src" / "deeply" / "nested"
        nested.mkdir(parents=True)

        result = find_project_index(nested)
        assert result == tmp_path / ".pysnippet" / "index.db"

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        # tmp_path itself has no .pysnippet, and its parents (real OS
        # dirs) shouldn't have one either in a test environment.
        result = find_project_index(tmp_path)
        assert result is None

    def test_defaults_to_cwd(self, tmp_path: Path, monkeypatch) -> None:
        index_dir = tmp_path / ".pysnippet"
        index_dir.mkdir()
        (index_dir / "index.db").write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = find_project_index()
        assert result == tmp_path / ".pysnippet" / "index.db"

    def test_nearest_index_wins_over_ancestor(self, tmp_path: Path) -> None:
        outer = tmp_path / ".pysnippet"
        outer.mkdir()
        (outer / "index.db").write_text("", encoding="utf-8")

        inner_project = tmp_path / "sub"
        inner_index = inner_project / ".pysnippet"
        inner_index.mkdir(parents=True)
        (inner_index / "index.db").write_text("", encoding="utf-8")

        result = find_project_index(inner_project)
        assert result == inner_project / ".pysnippet" / "index.db"


class TestBuildIndex:
    def test_indexes_python_files(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    return 1\n")
        _write(tmp_path / "b.py", "def bar():\n    return 2\n")

        result = build_index(tmp_path, embedder=_fake_embedder())

        assert result.files_scanned == 2
        assert result.snippets_indexed >= 2

    def test_skips_ignored_directories(self, tmp_path: Path) -> None:
        _write(tmp_path / "real.py", "def foo():\n    pass\n")
        _write(tmp_path / "node_modules" / "pkg.js", "function x() {}\n")

        result = build_index(tmp_path, embedder=_fake_embedder())
        assert result.files_scanned == 1

    def test_creates_db_at_default_location(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")

        result = build_index(tmp_path, embedder=_fake_embedder())
        assert result.db_path == tmp_path / ".pysnippet" / "index.db"
        assert result.db_path.exists()

    def test_custom_db_path(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        custom_path = tmp_path / "custom" / "my_index.db"

        result = build_index(tmp_path, embedder=_fake_embedder(), db_path=custom_path)
        assert result.db_path == custom_path
        assert custom_path.exists()

    def test_stores_snippets_retrievable_from_store(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    return 1\n")

        result = build_index(tmp_path, embedder=_fake_embedder())

        with SnippetStore(result.db_path) as store:
            assert store.count() == result.snippets_indexed
            snippets = store.all_snippets()
            assert any(s.name == "foo" for s in snippets)

    def test_stores_model_name_in_meta(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        embedder = _fake_embedder()

        result = build_index(tmp_path, embedder=embedder)

        with SnippetStore(result.db_path) as store:
            assert store.get_meta("model_name") == "fake-model"

    def test_stores_source_directory_in_meta(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")

        result = build_index(tmp_path, embedder=_fake_embedder())

        with SnippetStore(result.db_path) as store:
            assert store.get_meta("source_directory") == str(tmp_path.resolve())

    def test_empty_directory_produces_empty_index(self, tmp_path: Path) -> None:
        result = build_index(tmp_path, embedder=_fake_embedder())
        assert result.files_scanned == 0
        assert result.snippets_indexed == 0

        with SnippetStore(result.db_path) as store:
            assert store.count() == 0

    def test_rebuild_overwrites_previous_index(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        build_index(tmp_path, embedder=_fake_embedder())

        # Remove the file and rebuild -- old snippet should be gone
        (tmp_path / "a.py").unlink()
        _write(tmp_path / "b.py", "def bar():\n    pass\n")
        result = build_index(tmp_path, embedder=_fake_embedder())

        with SnippetStore(result.db_path) as store:
            names = {s.name for s in store.all_snippets()}
            assert "foo" not in names
            assert "bar" in names

    def test_relative_file_paths_use_forward_slashes(self, tmp_path: Path) -> None:
        _write(tmp_path / "sub" / "dir" / "a.py", "def foo():\n    pass\n")

        result = build_index(tmp_path, embedder=_fake_embedder())

        with SnippetStore(result.db_path) as store:
            snippets = store.all_snippets()
            assert any("sub/dir/a.py" == s.file_path for s in snippets)

    def test_multi_language_directory(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        _write(tmp_path / "b.js", "function bar() {\n  return 1;\n}\n")
        _write(tmp_path / "c.dart", "void main() {\n  print(1);\n}\n")

        result = build_index(tmp_path, embedder=_fake_embedder())
        assert result.files_scanned == 3

        with SnippetStore(result.db_path) as store:
            languages = {s.language for s in store.all_snippets()}
            assert languages == {"python", "javascript", "dart"}

    def test_default_embedder_used_when_none_given(self, tmp_path: Path, monkeypatch) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")

        created = {}

        class _FakeEmbeddingModel(EmbeddingModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created["instance"] = self
                self._model = _fake_embedder()._model

        monkeypatch.setattr("pysnippet_cli.indexer.EmbeddingModel", _FakeEmbeddingModel)

        result = build_index(tmp_path)
        assert result.snippets_indexed >= 1
        assert "instance" in created

    def test_string_directory_argument(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        result = build_index(str(tmp_path), embedder=_fake_embedder())
        assert result.files_scanned == 1

    def test_binary_file_with_valid_extension_is_skipped(self, tmp_path: Path) -> None:
        # walk_files only filters by extension, not content -- a binary
        # file masquerading as a .py file should be silently skipped
        # (read_text returns None) rather than crashing the index.
        _write(tmp_path / "real.py", "def foo():\n    pass\n")
        (tmp_path / "binary.py").write_bytes(b"\x00\x01\x02binary\xff")

        result = build_index(tmp_path, embedder=_fake_embedder())

        assert result.files_scanned == 1
        with SnippetStore(result.db_path) as store:
            assert all(s.file_path != "binary.py" for s in store.all_snippets())


def _touch(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


class TestUpdateIndex:
    def test_raises_without_existing_index(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            update_index(tmp_path, embedder=_fake_embedder())

    def test_unchanged_files_are_skipped(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        result = update_index(tmp_path, embedder=_fake_embedder(), db_path=build_result.db_path)

        assert result.files_unchanged == 1
        assert result.files_added == 0
        assert result.files_changed == 0
        assert result.snippets_indexed == 0

    def test_detects_new_file(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        _write(tmp_path / "b.py", "def bar():\n    pass\n")
        result = update_index(tmp_path, embedder=_fake_embedder(), db_path=build_result.db_path)

        assert result.files_added == 1
        assert result.files_unchanged == 1

        with SnippetStore(build_result.db_path) as store:
            names = {s.name for s in store.all_snippets()}
            assert names == {"foo", "bar"}

    def test_detects_changed_content(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.py"
        _write(file_path, "def foo():\n    return 1\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        original_mtime = file_path.stat().st_mtime
        _write(file_path, "def foo():\n    return 2\n")
        _touch(file_path, original_mtime + 100)  # ensure mtime differs

        result = update_index(tmp_path, embedder=_fake_embedder(), db_path=build_result.db_path)

        assert result.files_changed == 1
        assert result.files_added == 0

        with SnippetStore(build_result.db_path) as store:
            snippets = [s for s in store.all_snippets() if s.name == "foo"]
            assert len(snippets) == 1
            assert "return 2" in snippets[0].content

    def test_detects_removed_file(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        _write(tmp_path / "b.py", "def bar():\n    pass\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        (tmp_path / "b.py").unlink()
        result = update_index(tmp_path, embedder=_fake_embedder(), db_path=build_result.db_path)

        assert result.files_removed == 1

        with SnippetStore(build_result.db_path) as store:
            names = {s.name for s in store.all_snippets()}
            assert names == {"foo"}
            assert store.all_file_paths() == {"a.py"}

    def test_touch_without_content_change_is_unchanged(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.py"
        _write(file_path, "def foo():\n    pass\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        original_mtime = file_path.stat().st_mtime
        _touch(file_path, original_mtime + 100)  # mtime changes, content doesn't

        result = update_index(tmp_path, embedder=_fake_embedder(), db_path=build_result.db_path)

        assert result.files_changed == 0
        assert result.files_unchanged == 1
        assert result.snippets_indexed == 0

    def test_touch_refreshes_stored_mtime(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.py"
        _write(file_path, "def foo():\n    pass\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        new_mtime = file_path.stat().st_mtime + 100
        _touch(file_path, new_mtime)
        update_index(tmp_path, embedder=_fake_embedder(), db_path=build_result.db_path)

        with SnippetStore(build_result.db_path) as store:
            stored_mtime, _ = store.get_file_record("a.py")
            assert stored_mtime == pytest.approx(new_mtime)

    def test_unchanged_mtime_skips_reading_file(self, tmp_path: Path, monkeypatch) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        read_calls = []
        import pysnippet_cli.indexer as indexer_module

        original_read_text = indexer_module.read_text

        def _spy_read_text(path):
            read_calls.append(path)
            return original_read_text(path)

        monkeypatch.setattr(indexer_module, "read_text", _spy_read_text)

        update_index(tmp_path, embedder=_fake_embedder(), db_path=build_result.db_path)

        assert read_calls == []

    def test_snippets_indexed_counts_only_changed_and_added(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        _write(tmp_path / "b.py", "def bar():\n    pass\n\n\ndef baz():\n    pass\n")
        result = update_index(tmp_path, embedder=_fake_embedder(), db_path=build_result.db_path)

        # Only b.py's snippets should be counted -- a.py was unchanged
        assert result.snippets_indexed >= 2

        with SnippetStore(build_result.db_path) as store:
            assert store.count() >= 3  # foo + bar + baz

    def test_uses_model_name_from_existing_index_when_no_embedder_given(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        fake_model = MagicMock()
        fake_model.get_embedding_dimension.return_value = 4
        fake_model.encode.side_effect = lambda texts, **kw: np.random.rand(
            len(texts), 4
        ).astype(np.float32)

        load_calls = []

        def _fake_load_model(model_name):
            load_calls.append(model_name)
            return fake_model

        monkeypatch.setattr("pysnippet_cli.embedding._load_model", _fake_load_model)

        _write(tmp_path / "b.py", "def bar():\n    pass\n")
        update_index(tmp_path, db_path=build_result.db_path)

        assert load_calls == ["fake-model"]

    def test_string_directory_argument(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "def foo():\n    pass\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        result = update_index(
            str(tmp_path), embedder=_fake_embedder(), db_path=build_result.db_path
        )
        assert result.files_unchanged == 1

    def test_file_turned_binary_is_skipped(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.py"
        _write(file_path, "def foo():\n    pass\n")
        build_result = build_index(tmp_path, embedder=_fake_embedder())

        original_mtime = file_path.stat().st_mtime
        file_path.write_bytes(b"\x00\x01\x02binary\xff")
        _touch(file_path, original_mtime + 100)

        result = update_index(tmp_path, embedder=_fake_embedder(), db_path=build_result.db_path)

        # Not counted as changed (it was never re-embedded), and the
        # old snippet for a.py is still in the store untouched since we
        # never got far enough to delete/replace it.
        assert result.files_changed == 0
        assert result.files_added == 0
        with SnippetStore(build_result.db_path) as store:
            assert any(s.name == "foo" for s in store.all_snippets())
