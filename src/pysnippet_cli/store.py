"""Local persistence for indexed snippets and their embeddings.

Uses SQLite for storage. Embeddings are stored as raw float32 bytes in
a BLOB column rather than a separate numpy file, so the index stays a
single file with no risk of the two getting out of sync with each other.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from pysnippet_cli.snippet import Snippet

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snippets (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    kind TEXT NOT NULL,
    name TEXT,
    language TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snippets_file_path ON snippets(file_path);
"""

_SNIPPET_COLUMNS = "id, file_path, start_line, end_line, kind, name, language, content"


class SnippetStore:
    """A local SQLite-backed store of indexed snippets and their
    embedding vectors."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SnippetStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def add_snippets(self, snippets: list[Snippet], embeddings: np.ndarray) -> None:
        """Insert or update `snippets` with their corresponding
        `embeddings` (same length and order). Re-adding a snippet with
        an id that already exists overwrites that row."""
        if len(snippets) != len(embeddings):
            raise ValueError(
                f"snippets ({len(snippets)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        if not snippets:
            return

        rows = [
            (
                s.id,
                s.file_path,
                s.start_line,
                s.end_line,
                s.kind,
                s.name,
                s.language,
                s.content,
                np.asarray(embeddings[i], dtype=np.float32).tobytes(),
            )
            for i, s in enumerate(snippets)
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO snippets "
            f"({_SNIPPET_COLUMNS}, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM snippets").fetchone()
        return row[0]

    def get_snippet(self, snippet_id: str) -> Snippet | None:
        row = self._conn.execute(
            f"SELECT {_SNIPPET_COLUMNS} FROM snippets WHERE id = ?",
            (snippet_id,),
        ).fetchone()
        return _row_to_snippet(row) if row else None

    def delete_by_file(self, file_path: str) -> None:
        self._conn.execute("DELETE FROM snippets WHERE file_path = ?", (file_path,))
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM snippets")
        self._conn.execute("DELETE FROM meta")
        self._conn.commit()

    def all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Return (ids, embeddings) for every stored snippet, in a
        consistent order, for use by a similarity search index."""
        rows = self._conn.execute("SELECT id, embedding FROM snippets ORDER BY id").fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)

        ids = [row[0] for row in rows]
        vectors = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        return ids, vectors

    def all_snippets(self) -> list[Snippet]:
        rows = self._conn.execute(
            f"SELECT {_SNIPPET_COLUMNS} FROM snippets ORDER BY file_path, start_line"
        ).fetchall()
        return [_row_to_snippet(row) for row in rows]


def _row_to_snippet(row: tuple) -> Snippet:
    id_, file_path, start_line, end_line, kind, name, language, content = row
    return Snippet(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        kind=kind,
        name=name,
        language=language,
        id=id_,
    )
