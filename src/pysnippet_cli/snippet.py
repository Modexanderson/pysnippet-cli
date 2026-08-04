"""Data model for a single indexable code snippet."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Snippet:
    """A logical chunk of source code extracted from a file.

    `start_line`/`end_line` are 1-indexed and inclusive, matching how
    editors and `grep -n` report line numbers.
    """

    file_path: str
    start_line: int
    end_line: int
    content: str
    kind: str = "block"
    name: str | None = None
    language: str = "generic"
    id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", self._compute_id())

    def _compute_id(self) -> str:
        digest_input = f"{self.file_path}:{self.start_line}:{self.end_line}:{self.content}"
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        return digest[:12]

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def location(self) -> str:
        return f"{self.file_path}:{self.start_line}"
