"""Orchestrates the full indexing pipeline: walk -> parse -> embed -> store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pysnippet_cli.embedding import EmbeddingModel
from pysnippet_cli.parsers import parse
from pysnippet_cli.snippet import Snippet
from pysnippet_cli.splitter import read_text
from pysnippet_cli.store import SnippetStore
from pysnippet_cli.walker import language_for, walk_files

INDEX_DIR_NAME = ".pysnippet"
INDEX_FILE_NAME = "index.db"


def index_path_for(directory: Path | str) -> Path:
    """The default index location for a project directory: a sibling
    `.pysnippet/index.db`, analogous to how `.git` sits inside a repo."""
    return Path(directory) / INDEX_DIR_NAME / INDEX_FILE_NAME


@dataclass
class IndexResult:
    files_scanned: int
    snippets_indexed: int
    db_path: Path


def build_index(
    directory: Path | str,
    *,
    embedder: EmbeddingModel | None = None,
    db_path: Path | str | None = None,
    batch_size: int = 32,
) -> IndexResult:
    """Walk `directory`, extract snippets from every source file, embed
    them, and store the result in a local SQLite index.

    Overwrites any existing index at the target path -- this is a full
    rebuild, not incremental (see the `update` command for that).
    """
    directory = Path(directory).resolve()
    embedder = embedder or EmbeddingModel()
    resolved_db_path = Path(db_path) if db_path is not None else index_path_for(directory)

    all_snippets: list[Snippet] = []
    files_scanned = 0

    for file_path in walk_files(directory):
        language = language_for(file_path)
        if language is None:
            continue
        content = read_text(file_path)
        if content is None:
            continue

        rel_path = file_path.relative_to(directory).as_posix()
        all_snippets.extend(parse(content, file_path=rel_path, language=language))
        files_scanned += 1

    with SnippetStore(resolved_db_path) as store:
        store.clear()
        if all_snippets:
            embeddings = embedder.embed_snippets(all_snippets, batch_size=batch_size)
            store.add_snippets(all_snippets, embeddings)
        store.set_meta("model_name", embedder.model_name)
        store.set_meta("source_directory", str(directory))

    return IndexResult(
        files_scanned=files_scanned,
        snippets_indexed=len(all_snippets),
        db_path=resolved_db_path,
    )
