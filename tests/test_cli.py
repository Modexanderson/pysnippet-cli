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


class TestFindCommand:
    def test_stub_exits_nonzero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["find", "parse json"])
        assert result.exit_code == 1
        assert "parse json" in result.output

    def test_requires_query_argument(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["find"])
        assert result.exit_code != 0

    def test_top_k_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["find", "test query", "--top-k", "10"])
        assert "top_k: 10" in result.output

    def test_top_k_default(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["find", "test query"])
        assert "top_k: 5" in result.output


class TestShowCommand:
    def test_stub_exits_nonzero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["show", "abc123"])
        assert result.exit_code == 1
        assert "abc123" in result.output

    def test_requires_id_argument(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["show"])
        assert result.exit_code != 0


class TestUpdateCommand:
    def test_stub_exits_nonzero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["update"])
        assert result.exit_code == 1
