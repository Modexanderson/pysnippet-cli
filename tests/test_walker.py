from pathlib import Path

import pytest

from pysnippet_cli.walker import language_for, walk_files


def _touch(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestWalkFiles:
    def test_finds_source_files(self, tmp_path: Path) -> None:
        _touch(tmp_path / "a.py")
        _touch(tmp_path / "b.js")
        _touch(tmp_path / "readme.txt")

        found = {p.name for p in walk_files(tmp_path)}
        assert found == {"a.py", "b.js"}

    def test_recurses_into_subdirectories(self, tmp_path: Path) -> None:
        _touch(tmp_path / "top.py")
        _touch(tmp_path / "nested" / "deep" / "bottom.py")

        found = {p.name for p in walk_files(tmp_path)}
        assert found == {"top.py", "bottom.py"}

    def test_skips_default_ignore_dirs(self, tmp_path: Path) -> None:
        _touch(tmp_path / "real.py")
        _touch(tmp_path / "node_modules" / "pkg" / "index.js")
        _touch(tmp_path / ".git" / "hooks" / "pre-commit.py")
        _touch(tmp_path / "__pycache__" / "cached.py")
        _touch(tmp_path / ".venv" / "lib" / "site.py")

        found = {p.name for p in walk_files(tmp_path)}
        assert found == {"real.py"}

    def test_skips_hidden_directories(self, tmp_path: Path) -> None:
        _touch(tmp_path / "visible.py")
        _touch(tmp_path / ".hidden" / "secret.py")

        found = {p.name for p in walk_files(tmp_path)}
        assert found == {"visible.py"}

    def test_skips_files_over_size_limit(self, tmp_path: Path) -> None:
        small = tmp_path / "small.py"
        big = tmp_path / "big.py"
        _touch(small, "x = 1\n")
        _touch(big, "x = 1\n" * 100)

        found = {p.name for p in walk_files(tmp_path, max_file_size=200)}
        assert found == {"small.py"}

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert list(walk_files(tmp_path)) == []

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            list(walk_files(tmp_path / "does-not-exist"))

    def test_file_instead_of_directory_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.py"
        _touch(f)
        with pytest.raises(NotADirectoryError):
            list(walk_files(f))

    def test_custom_extensions(self, tmp_path: Path) -> None:
        _touch(tmp_path / "a.py")
        _touch(tmp_path / "b.custom")

        found = {p.name for p in walk_files(tmp_path, extensions={".custom": "custom"})}
        assert found == {"b.custom"}

    def test_custom_ignore_dirs(self, tmp_path: Path) -> None:
        _touch(tmp_path / "keep.py")
        _touch(tmp_path / "skip_me" / "hidden.py")

        found = {p.name for p in walk_files(tmp_path, ignore_dirs={"skip_me"})}
        assert found == {"keep.py"}

    def test_results_are_sorted_deterministically(self, tmp_path: Path) -> None:
        for name in ["z.py", "a.py", "m.py"]:
            _touch(tmp_path / name)

        names = [p.name for p in walk_files(tmp_path)]
        assert names == sorted(names)

    def test_multiple_languages_covered(self, tmp_path: Path) -> None:
        for name in ["a.py", "b.ts", "c.dart", "d.go", "e.rs"]:
            _touch(tmp_path / name)

        found = {p.name for p in walk_files(tmp_path)}
        assert found == {"a.py", "b.ts", "c.dart", "d.go", "e.rs"}


class TestLanguageFor:
    def test_known_extension(self) -> None:
        assert language_for(Path("foo.py")) == "python"
        assert language_for(Path("foo.ts")) == "typescript"
        assert language_for(Path("foo.dart")) == "dart"

    def test_unknown_extension(self) -> None:
        assert language_for(Path("foo.xyz")) is None

    def test_custom_extension_map(self) -> None:
        assert language_for(Path("foo.custom"), {".custom": "custom-lang"}) == "custom-lang"
