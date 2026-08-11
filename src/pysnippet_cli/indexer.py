"""Orchestrates the full indexing pipeline: walk -> parse -> embed -> store.

Also handles incremental updates: each indexed file's mtime and content
hash are recorded, so a later `update_index()` call can skip files that
haven't changed and only re-embed the ones that have.
"""

from __future__ import annotations

import hashlib
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


def find_project_index(start: Path | str | None = None) -> Path | None:
    """Walk upward from `start` (default: the current directory) looking
    for a `.pysnippet/index.db`, the same way git locates `.git` from
    any subdirectory of a repo. Returns None if none is found before
    reaching the filesystem root."""
    current = Path(start).resolve() if start is not None else Path.cwd()
    for candidate in (current, *current.parents):
        db_path = index_path_for(candidate)
        if db_path.exists():
            return db_path
    return None


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class _ScannedFile:
    rel_path: str
    language: str
    content: str
    mtime: float
    content_hash: str


def _scan_file(file_path: Path, directory: Path) -> _ScannedFile | None:
    """Read and hash a single source file, or None if it should be
    skipped (unsupported extension, binary, or unreadable)."""
    language = language_for(file_path)
    if language is None:
        return None
    content = read_text(file_path)
    if content is None:
        return None

    rel_path = file_path.relative_to(directory).as_posix()
    mtime = file_path.stat().st_mtime
    return _ScannedFile(rel_path, language, content, mtime, _content_hash(content))


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
    rebuild. Records each file's mtime/content hash so a later
    `update_index()` call can skip unchanged files.
    """
    directory = Path(directory).resolve()
    embedder = embedder or EmbeddingModel()
    resolved_db_path = Path(db_path) if db_path is not None else index_path_for(directory)

    all_snippets: list[Snippet] = []
    scanned_files: list[_ScannedFile] = []

    for file_path in walk_files(directory):
        scanned = _scan_file(file_path, directory)
        if scanned is None:
            continue
        scanned_files.append(scanned)
        all_snippets.extend(
            parse(scanned.content, file_path=scanned.rel_path, language=scanned.language)
        )

    with SnippetStore(resolved_db_path) as store:
        store.clear()
        if all_snippets:
            embeddings = embedder.embed_snippets(all_snippets, batch_size=batch_size)
            store.add_snippets(all_snippets, embeddings)
        for scanned in scanned_files:
            store.set_file_record(scanned.rel_path, scanned.mtime, scanned.content_hash)
        store.set_meta("model_name", embedder.model_name)
        store.set_meta("source_directory", str(directory))

    return IndexResult(
        files_scanned=len(scanned_files),
        snippets_indexed=len(all_snippets),
        db_path=resolved_db_path,
    )


@dataclass
class UpdateResult:
    files_added: int
    files_changed: int
    files_removed: int
    files_unchanged: int
    snippets_indexed: int
    db_path: Path


def update_index(
    directory: Path | str,
    *,
    embedder: EmbeddingModel | None = None,
    db_path: Path | str | None = None,
    batch_size: int = 32,
) -> UpdateResult:
    """Incrementally re-index `directory`: only files that are new or
    whose content changed since the last index/update are re-embedded.
    Files removed from disk have their snippets removed too.

    Uses a two-tier check per file: if its mtime matches what's stored,
    it's assumed unchanged and never even read (the fast path -- this
    is what makes incremental updates cheap on large trees). If the
    mtime differs, the file is read and its content hash compared
    against what's stored; only a genuine hash mismatch triggers
    re-embedding, so a `touch` or a git checkout that bumps mtimes
    without changing content doesn't trigger unnecessary work (it does
    still cost a read+hash, just not a re-embed).

    Requires an existing index at the target path (built via
    `build_index`) -- raises FileNotFoundError otherwise.
    """
    directory = Path(directory).resolve()
    resolved_db_path = Path(db_path) if db_path is not None else index_path_for(directory)

    if not resolved_db_path.exists():
        raise FileNotFoundError(
            f"No index found at {resolved_db_path}. Run `pysnippet index` first."
        )

    with SnippetStore(resolved_db_path) as store:
        stored_model_name = store.get_meta("model_name")
        if embedder is None:
            embedder = (
                EmbeddingModel(model_name=stored_model_name)
                if stored_model_name
                else EmbeddingModel()
            )

        known_files = store.all_file_paths()
        seen_files: set[str] = set()
        pending_snippets: list[Snippet] = []
        pending_records: list[_ScannedFile] = []
        files_added = 0
        files_changed = 0
        files_unchanged = 0

        for file_path in walk_files(directory):
            language = language_for(file_path)
            if language is None:
                continue

            rel_path = file_path.relative_to(directory).as_posix()
            seen_files.add(rel_path)
            existing = store.get_file_record(rel_path)

            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                continue

            # Fast path: mtime matches what we recorded -- trust that
            # content hasn't changed without reading the file at all.
            if existing is not None and existing[0] == mtime:
                files_unchanged += 1
                continue

            content = read_text(file_path)
            if content is None:
                continue
            content_hash = _content_hash(content)

            # mtime changed but content didn't (touch, git checkout,
            # etc.) -- refresh the stored mtime so the fast path kicks
            # in next time, but skip re-embedding.
            if existing is not None and existing[1] == content_hash:
                files_unchanged += 1
                store.set_file_record(rel_path, mtime, content_hash)
                continue

            files_added += 1 if existing is None else 0
            files_changed += 1 if existing is not None else 0

            store.delete_by_file(rel_path)
            pending_snippets.extend(parse(content, file_path=rel_path, language=language))
            pending_records.append(_ScannedFile(rel_path, language, content, mtime, content_hash))

        removed_files = known_files - seen_files
        for rel_path in removed_files:
            store.delete_by_file(rel_path)
            store.delete_file_record(rel_path)

        if pending_snippets:
            embeddings = embedder.embed_snippets(pending_snippets, batch_size=batch_size)
            store.add_snippets(pending_snippets, embeddings)

        for scanned in pending_records:
            store.set_file_record(scanned.rel_path, scanned.mtime, scanned.content_hash)

        store.set_meta("model_name", embedder.model_name)
        store.set_meta("source_directory", str(directory))

    return UpdateResult(
        files_added=files_added,
        files_changed=files_changed,
        files_removed=len(removed_files),
        files_unchanged=files_unchanged,
        snippets_indexed=len(pending_snippets),
        db_path=resolved_db_path,
    )
