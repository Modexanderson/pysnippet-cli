from unittest.mock import MagicMock, patch

import numpy as np
from click.testing import CliRunner

from pysnippet_cli import __version__
from pysnippet_cli.cli import main


def _fake_model(dim: int = 4) -> MagicMock:
    model = MagicMock()
    model.get_embedding_dimension.return_value = dim

    def _encode(texts, **kwargs):
        return np.random.rand(len(texts), dim).astype(np.float32)

    model.encode.side_effect = _encode
    return model


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "find code by meaning" in result.output


def test_help_lists_all_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    for command in ("index", "find", "show", "update"):
        assert command in result.output


class TestIndexCommand:
    @patch("pysnippet_cli.embedding._load_model")
    def test_indexes_directory(self, mock_load, tmp_path) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert result.exit_code == 0
        assert "Indexed" in result.output
        assert (tmp_path / ".pysnippet" / "index.db").exists()

    @patch("pysnippet_cli.embedding._load_model")
    def test_empty_directory_exits_nonzero(self, mock_load, tmp_path) -> None:
        mock_load.return_value = _fake_model()

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert result.exit_code == 1
        assert "No indexable source files" in result.output

    def test_rejects_nonexistent_directory(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["index", "/no/such/directory"])
        assert result.exit_code != 0

    @patch("pysnippet_cli.embedding._load_model")
    def test_defaults_to_current_directory(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["index"])

        assert result.exit_code == 0

    @patch("pysnippet_cli.embedding._load_model")
    def test_model_option_passed_through(self, mock_load, tmp_path) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path), "--model", "custom-model"])

        assert result.exit_code == 0
        mock_load.assert_called_once_with("custom-model")


class TestConfigIntegration:
    @patch("pysnippet_cli.embedding._load_model")
    def test_index_uses_model_from_config(self, mock_load, tmp_path) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (tmp_path / ".pysnippetrc").write_text('model = "config-model"\n', encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert result.exit_code == 0
        mock_load.assert_called_once_with("config-model")

    @patch("pysnippet_cli.embedding._load_model")
    def test_explicit_model_option_overrides_config(self, mock_load, tmp_path) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (tmp_path / ".pysnippetrc").write_text('model = "config-model"\n', encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path), "--model", "cli-model"])

        assert result.exit_code == 0
        mock_load.assert_called_once_with("cli-model")

    @patch("pysnippet_cli.embedding._load_model")
    def test_index_respects_ignore_patterns_from_config(self, mock_load, tmp_path) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "real.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (tmp_path / "skip.generated.py").write_text("def bar():\n    pass\n", encoding="utf-8")
        (tmp_path / ".pysnippetrc").write_text(
            'ignore = ["*.generated.py"]\n', encoding="utf-8"
        )

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        from pysnippet_cli.indexer import find_project_index
        from pysnippet_cli.store import SnippetStore

        db_path = find_project_index(tmp_path)
        with SnippetStore(db_path) as store:
            names = {s.name for s in store.all_snippets()}
            assert names == {"foo"}

    @patch("pysnippet_cli.embedding._load_model")
    def test_index_respects_language_filter_from_config(self, mock_load, tmp_path) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (tmp_path / "b.js").write_text("function bar() {\n  return 1;\n}\n", encoding="utf-8")
        (tmp_path / ".pysnippetrc").write_text('languages = ["python"]\n', encoding="utf-8")

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        from pysnippet_cli.indexer import find_project_index
        from pysnippet_cli.store import SnippetStore

        db_path = find_project_index(tmp_path)
        with SnippetStore(db_path) as store:
            languages = {s.language for s in store.all_snippets()}
            assert languages == {"python"}

    @patch("pysnippet_cli.embedding._load_model")
    def test_find_uses_top_k_from_config(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(
                f"def func{i}():\n    return {i}\n", encoding="utf-8"
            )
        (tmp_path / ".pysnippetrc").write_text("top_k = 2\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        result = runner.invoke(main, ["find", "function"])
        assert result.exit_code == 0
        assert result.output.count("id=") == 2

    @patch("pysnippet_cli.embedding._load_model")
    def test_explicit_top_k_overrides_config(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(
                f"def func{i}():\n    return {i}\n", encoding="utf-8"
            )
        (tmp_path / ".pysnippetrc").write_text("top_k = 2\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        result = runner.invoke(main, ["find", "function", "--top-k", "4"])
        assert result.exit_code == 0
        assert result.output.count("id=") == 4

    @patch("pysnippet_cli.embedding._load_model")
    def test_update_respects_ignore_patterns_from_config(
        self, mock_load, tmp_path, monkeypatch
    ) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "real.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (tmp_path / ".pysnippetrc").write_text(
            'ignore = ["*.generated.py"]\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        (tmp_path / "skip.generated.py").write_text("def bar():\n    pass\n", encoding="utf-8")
        result = runner.invoke(main, ["update"])

        assert result.exit_code == 0
        assert "Added 0" in result.output


class TestFindCommand:
    def test_requires_query_argument(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["find"])
        assert result.exit_code != 0

    @patch("pysnippet_cli.embedding._load_model")
    def test_no_index_found(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["find", "some query"])

        assert result.exit_code == 1
        assert "No index found" in result.output

    @patch("pysnippet_cli.embedding._load_model")
    def test_empty_index(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        result = runner.invoke(main, ["find", "some query"])
        assert result.exit_code == 1
        assert "Index is empty" in result.output

    @patch("pysnippet_cli.embedding._load_model")
    def test_returns_results(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        index_result = runner.invoke(main, ["index", str(tmp_path)])
        assert index_result.exit_code == 0

        result = runner.invoke(main, ["find", "foo function"])
        assert result.exit_code == 0
        assert "foo" in result.output
        assert "a.py" in result.output
        assert "id=" in result.output

    @patch("pysnippet_cli.embedding._load_model")
    def test_top_k_limits_results(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(
                f"def func{i}():\n    return {i}\n", encoding="utf-8"
            )
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        result = runner.invoke(main, ["find", "function", "--top-k", "2"])
        assert result.exit_code == 0
        assert result.output.count("id=") == 2

    @patch("pysnippet_cli.embedding._load_model")
    def test_finds_index_from_subdirectory(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)

        result = runner.invoke(main, ["find", "foo"])
        assert result.exit_code == 0
        assert "id=" in result.output

    @patch("pysnippet_cli.embedding._load_model")
    def test_uses_model_stored_in_index_meta(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path), "--model", "custom-model"])
        mock_load.reset_mock()

        runner.invoke(main, ["find", "foo"])
        mock_load.assert_called_once_with("custom-model")


class TestShowCommand:
    def test_requires_id_argument(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["show"])
        assert result.exit_code != 0

    @patch("pysnippet_cli.embedding._load_model")
    def test_no_index_found(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["show", "abc123"])

        assert result.exit_code == 1
        assert "No index found" in result.output

    @patch("pysnippet_cli.embedding._load_model")
    def test_snippet_not_found(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        result = runner.invoke(main, ["show", "does-not-exist"])
        assert result.exit_code == 1
        assert "No snippet found" in result.output

    @patch("pysnippet_cli.embedding._load_model")
    def test_shows_snippet_content(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        from pysnippet_cli.indexer import find_project_index
        from pysnippet_cli.store import SnippetStore

        db_path = find_project_index(tmp_path)
        with SnippetStore(db_path) as store:
            snippet = next(s for s in store.all_snippets() if s.name == "foo")

        result = runner.invoke(main, ["show", snippet.id])
        assert result.exit_code == 0
        assert "return 42" in result.output
        assert "foo" in result.output


class TestUpdateCommand:
    @patch("pysnippet_cli.embedding._load_model")
    def test_no_index_found(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["update"])

        assert result.exit_code == 1
        assert "No index found" in result.output

    @patch("pysnippet_cli.embedding._load_model")
    def test_updates_with_new_file(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        (tmp_path / "b.py").write_text("def bar():\n    pass\n", encoding="utf-8")
        result = runner.invoke(main, ["update"])

        assert result.exit_code == 0
        assert "Added 1" in result.output

    @patch("pysnippet_cli.embedding._load_model")
    def test_no_changes_reports_unchanged(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        result = runner.invoke(main, ["update"])
        assert result.exit_code == 0
        assert "unchanged 1" in result.output

    @patch("pysnippet_cli.embedding._load_model")
    def test_removed_file_is_reflected(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("def bar():\n    pass\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        (tmp_path / "b.py").unlink()
        result = runner.invoke(main, ["update"])

        assert result.exit_code == 0
        assert "removed 1" in result.output

    @patch("pysnippet_cli.embedding._load_model")
    def test_works_from_subdirectory(self, mock_load, tmp_path, monkeypatch) -> None:
        mock_load.return_value = _fake_model()
        (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")

        runner = CliRunner()
        runner.invoke(main, ["index", str(tmp_path)])

        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)

        result = runner.invoke(main, ["update"])
        assert result.exit_code == 0
