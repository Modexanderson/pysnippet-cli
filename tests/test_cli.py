from click.testing import CliRunner

from pysnippet_cli import __version__
from pysnippet_cli.cli import main


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
    def test_stub_exits_nonzero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["index", "."])
        assert result.exit_code == 1
        assert "not yet implemented" in result.output

    def test_rejects_nonexistent_directory(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["index", "/no/such/directory"])
        assert result.exit_code != 0

    def test_defaults_to_current_directory(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["index"])
        assert result.exit_code == 1


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
