from pathlib import Path

from pysnippet_cli.config import (
    DEFAULT_TOP_K,
    Config,
    find_config_file,
    load_config,
    load_config_file,
)


def _write_config(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestFindConfigFile:
    def test_finds_config_in_start_directory(self, tmp_path: Path) -> None:
        _write_config(tmp_path / ".pysnippetrc", "")
        assert find_config_file(tmp_path) == tmp_path / ".pysnippetrc"

    def test_finds_config_in_parent_directory(self, tmp_path: Path) -> None:
        _write_config(tmp_path / ".pysnippetrc", "")
        nested = tmp_path / "src" / "deeply" / "nested"
        nested.mkdir(parents=True)

        assert find_config_file(nested) == tmp_path / ".pysnippetrc"

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        assert find_config_file(tmp_path) is None

    def test_defaults_to_cwd(self, tmp_path: Path, monkeypatch) -> None:
        _write_config(tmp_path / ".pysnippetrc", "")
        monkeypatch.chdir(tmp_path)
        assert find_config_file() == tmp_path / ".pysnippetrc"

    def test_nearest_config_wins_over_ancestor(self, tmp_path: Path) -> None:
        _write_config(tmp_path / ".pysnippetrc", "model = 'outer-model'")
        inner = tmp_path / "sub"
        inner.mkdir()
        _write_config(inner / ".pysnippetrc", "model = 'inner-model'")

        result = find_config_file(inner)
        assert result == inner / ".pysnippetrc"


class TestLoadConfigFile:
    """load_config_file() loads a known, already-located path directly."""

    def test_nonexistent_file_returns_defaults(self, tmp_path: Path) -> None:
        config = load_config_file(tmp_path / "nonexistent.pysnippetrc")
        assert config == Config()

    def test_loads_model(self, tmp_path: Path) -> None:
        path = tmp_path / ".pysnippetrc"
        _write_config(path, 'model = "custom-model"\n')
        config = load_config_file(path)
        assert config.model == "custom-model"

    def test_loads_top_k(self, tmp_path: Path) -> None:
        path = tmp_path / ".pysnippetrc"
        _write_config(path, "top_k = 10\n")
        config = load_config_file(path)
        assert config.top_k == 10

    def test_missing_top_k_uses_default(self, tmp_path: Path) -> None:
        path = tmp_path / ".pysnippetrc"
        _write_config(path, 'model = "custom-model"\n')
        config = load_config_file(path)
        assert config.top_k == DEFAULT_TOP_K

    def test_loads_ignore_patterns(self, tmp_path: Path) -> None:
        path = tmp_path / ".pysnippetrc"
        _write_config(path, 'ignore = ["*.generated.py", "vendor/"]\n')
        config = load_config_file(path)
        assert config.ignore == ["*.generated.py", "vendor/"]

    def test_missing_ignore_defaults_to_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / ".pysnippetrc"
        _write_config(path, 'model = "custom-model"\n')
        config = load_config_file(path)
        assert config.ignore == []

    def test_loads_languages(self, tmp_path: Path) -> None:
        path = tmp_path / ".pysnippetrc"
        _write_config(path, 'languages = ["python", "typescript"]\n')
        config = load_config_file(path)
        assert config.languages == ["python", "typescript"]

    def test_missing_languages_means_no_filter(self, tmp_path: Path) -> None:
        path = tmp_path / ".pysnippetrc"
        _write_config(path, 'model = "custom-model"\n')
        config = load_config_file(path)
        assert config.languages is None

    def test_full_config(self, tmp_path: Path) -> None:
        path = tmp_path / ".pysnippetrc"
        _write_config(
            path,
            """
model = "all-MiniLM-L6-v2"
top_k = 8
ignore = ["*.generated.py", "vendor/", "tests/fixtures/*"]
languages = ["python", "javascript", "typescript"]
""",
        )
        config = load_config_file(path)
        assert config.model == "all-MiniLM-L6-v2"
        assert config.top_k == 8
        assert config.ignore == ["*.generated.py", "vendor/", "tests/fixtures/*"]
        assert config.languages == ["python", "javascript", "typescript"]

    def test_empty_config_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".pysnippetrc"
        _write_config(path, "")
        config = load_config_file(path)
        assert config == Config()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        path = tmp_path / ".pysnippetrc"
        _write_config(path, 'model = "custom-model"\n')
        config = load_config_file(str(path))
        assert config.model == "custom-model"


class TestLoadConfig:
    """load_config() searches upward from a start directory, like
    find_config_file/find_project_index -- it never takes a file path
    directly (use load_config_file for that)."""

    def test_no_config_file_returns_defaults(self, tmp_path: Path) -> None:
        config = load_config(tmp_path)
        assert config == Config()

    def test_defaults_to_cwd_when_no_start_given(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config == Config()

    def test_finds_and_loads_config_in_start_directory(self, tmp_path: Path) -> None:
        _write_config(tmp_path / ".pysnippetrc", 'model = "custom-model"\n')
        config = load_config(tmp_path)
        assert config.model == "custom-model"

    def test_finds_and_loads_config_in_parent_directory(self, tmp_path: Path) -> None:
        _write_config(tmp_path / ".pysnippetrc", 'model = "custom-model"\n')
        nested = tmp_path / "src" / "deeply" / "nested"
        nested.mkdir(parents=True)

        config = load_config(nested)
        assert config.model == "custom-model"

    def test_auto_discovers_from_cwd_when_no_start_given(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_config(tmp_path / ".pysnippetrc", 'model = "discovered-model"\n')
        monkeypatch.chdir(tmp_path)

        config = load_config()
        assert config.model == "discovered-model"

    def test_full_config_via_search(self, tmp_path: Path) -> None:
        _write_config(
            tmp_path / ".pysnippetrc",
            """
model = "all-MiniLM-L6-v2"
top_k = 8
ignore = ["*.generated.py", "vendor/"]
languages = ["python", "dart"]
""",
        )
        config = load_config(tmp_path)
        assert config.model == "all-MiniLM-L6-v2"
        assert config.top_k == 8
        assert config.ignore == ["*.generated.py", "vendor/"]
        assert config.languages == ["python", "dart"]

    def test_accepts_string_start(self, tmp_path: Path) -> None:
        _write_config(tmp_path / ".pysnippetrc", 'model = "custom-model"\n')
        config = load_config(str(tmp_path))
        assert config.model == "custom-model"
