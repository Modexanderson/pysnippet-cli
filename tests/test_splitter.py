from pathlib import Path

from pysnippet_cli.splitter import read_text, split_generic


class TestReadText:
    def test_reads_utf8_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_bytes(b"x = 1\n")
        assert read_text(f) == "x = 1\n"

    def test_reads_utf8_bom(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_bytes(b"\xef\xbb\xbfx = 1\n")
        assert read_text(f) == "x = 1\n"

    def test_reads_latin1_fallback(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_bytes("caf\xe9 = 1\n".encode("latin-1"))
        result = read_text(f)
        assert result is not None
        assert "caf" in result

    def test_returns_none_for_binary(self, tmp_path: Path) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"\x00\x01\x02\xff")
        assert read_text(f) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert read_text(tmp_path / "missing.py") is None


class TestSplitGeneric:
    def test_empty_content_returns_no_snippets(self) -> None:
        assert split_generic("", file_path="a.py") == []

    def test_single_block_no_blank_lines(self) -> None:
        content = "def foo():\n    return 1\n"
        snippets = split_generic(content, file_path="a.py", min_lines=1)
        assert len(snippets) == 1
        assert snippets[0].start_line == 1
        assert snippets[0].end_line == 2

    def test_splits_on_blank_lines(self) -> None:
        content = "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n"
        snippets = split_generic(content, file_path="a.py", min_lines=1)
        assert len(snippets) == 2
        assert "foo" in snippets[0].content
        assert "bar" in snippets[1].content

    def test_line_numbers_are_one_indexed_and_correct(self) -> None:
        content = "a\nb\n\nc\nd\n"
        snippets = split_generic(content, file_path="a.py", min_lines=1)
        assert snippets[0].start_line == 1
        assert snippets[0].end_line == 2
        assert snippets[1].start_line == 4
        assert snippets[1].end_line == 5

    def test_merges_small_blocks_into_next(self) -> None:
        # First block is 1 line (below min_lines=3), should merge forward
        content = "x = 1\n\ndef foo():\n    a = 1\n    b = 2\n"
        snippets = split_generic(content, file_path="a.py", min_lines=3)
        assert len(snippets) == 1
        assert "x = 1" in snippets[0].content
        assert "foo" in snippets[0].content

    def test_merges_trailing_small_block_backward(self) -> None:
        content = "def foo():\n    a = 1\n    b = 2\n\nx = 1\n"
        snippets = split_generic(content, file_path="a.py", min_lines=3)
        assert len(snippets) == 1
        assert "foo" in snippets[0].content
        assert "x = 1" in snippets[0].content

    def test_all_small_blocks_merge_into_one(self) -> None:
        content = "a\n\nb\n\nc\n"
        snippets = split_generic(content, file_path="a.py", min_lines=3)
        assert len(snippets) == 1

    def test_caps_oversized_blocks_at_max_lines(self) -> None:
        lines = [f"line{i}" for i in range(150)]
        content = "\n".join(lines)
        snippets = split_generic(content, file_path="a.py", max_lines=60, min_lines=1)
        assert len(snippets) == 3  # 60 + 60 + 30
        assert snippets[0].line_count == 60
        assert snippets[1].line_count == 60
        assert snippets[2].line_count == 30

    def test_chunk_line_numbers_are_contiguous(self) -> None:
        lines = [f"line{i}" for i in range(10)]
        content = "\n".join(lines)
        snippets = split_generic(content, file_path="a.py", max_lines=4, min_lines=1)
        assert [(s.start_line, s.end_line) for s in snippets] == [
            (1, 4),
            (5, 8),
            (9, 10),
        ]

    def test_file_path_and_language_propagate(self) -> None:
        snippets = split_generic(
            "x = 1\n", file_path="src/foo.py", language="python", min_lines=1
        )
        assert snippets[0].file_path == "src/foo.py"
        assert snippets[0].language == "python"

    def test_leading_and_trailing_blank_lines_ignored(self) -> None:
        content = "\n\ndef foo():\n    pass\n\n\n"
        snippets = split_generic(content, file_path="a.py", min_lines=1)
        assert len(snippets) == 1
        assert snippets[0].start_line == 3

    def test_only_blank_lines_returns_no_snippets(self) -> None:
        content = "\n\n\n"
        snippets = split_generic(content, file_path="a.py", min_lines=1)
        assert snippets == []

    def test_content_reconstructs_original_lines(self) -> None:
        content = "def foo():\n    return 1\n"
        snippets = split_generic(content, file_path="a.py", min_lines=1)
        assert snippets[0].content == "def foo():\n    return 1"
