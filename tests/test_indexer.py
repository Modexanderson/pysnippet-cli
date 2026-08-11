from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from pysnippet_cli.embedding import EmbeddingModel
from pysnippet_cli.indexer import build_index, find_project_index, index_path_for
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
