"""Project configuration loaded from a `.pysnippetrc` (TOML) file.

Located the same way the index itself is -- walking upward from the
current directory, the same as git finds `.git`. Values here act as
defaults: an explicit CLI option always overrides the config file.

Example `.pysnippetrc`:

    model = "all-MiniLM-L6-v2"
    top_k = 5
    ignore = ["*.generated.py", "vendor/", "tests/fixtures/*"]
    languages = ["python", "javascript", "typescript"]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CONFIG_FILE_NAME = ".pysnippetrc"
DEFAULT_TOP_K = 5


@dataclass
class Config:
    model: str | None = None
    top_k: int = DEFAULT_TOP_K
    ignore: list[str] = field(default_factory=list)
    languages: list[str] | None = None  # None means no filter -- all supported languages


def find_config_file(start: Path | str | None = None) -> Path | None:
    """Walk upward from `start` (default: the current directory) looking
    for a `.pysnippetrc`. Returns None if none is found before reaching
    the filesystem root."""
    current = Path(start).resolve() if start is not None else Path.cwd()
    for candidate in (current, *current.parents):
        config_path = candidate / CONFIG_FILE_NAME
        if config_path.exists():
            return config_path
    return None


def load_config(start: Path | str | None = None) -> Config:
    """Load config by searching for a `.pysnippetrc` starting from
    `start` (default: the current directory) and walking upward -- the
    same search `find_config_file` (and `find_project_index`) use.
    Returns defaults if none is found."""
    config_path = find_config_file(start)
    if config_path is None:
        return Config()
    return load_config_file(config_path)


def load_config_file(config_path: Path | str) -> Config:
    """Load config from a known `.pysnippetrc` path directly, with no
    search involved. Returns defaults if the file doesn't exist."""
    config_path = Path(config_path)
    if not config_path.exists():
        return Config()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    return Config(
        model=data.get("model"),
        top_k=data.get("top_k", DEFAULT_TOP_K),
        ignore=list(data.get("ignore", [])),
        languages=list(data["languages"]) if "languages" in data else None,
    )
