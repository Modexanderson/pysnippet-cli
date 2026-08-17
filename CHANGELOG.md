## 1.0.0

Initial release.

- **CLI**: `pysnippet index`, `find`, `show`, `update` (`click`-based)
- **File walker**: recursive scan with sensible default ignores
  (`.git`, `node_modules`, `.venv`, build output, etc.), configurable
  `ignore_patterns` (glob-style) and `languages` filter
- **Language-aware parsing**:
  - Python via `ast` — top-level functions/classes plus one level of
    methods (`Class.method`), decorator-inclusive spans
  - JavaScript/TypeScript/Dart via regex + brace matching — functions,
    classes, arrow functions, methods, with control-keyword filtering
    to avoid false positives on `if`/`for`/`while`/etc.
  - Generic blank-line-delimited block splitter as a fallback for
    every other language, so nothing is ever dropped
- **Embeddings**: local, on-device via `sentence-transformers`
  (default model `all-MiniLM-L6-v2`), lazily loaded so `--help` stays
  instant
- **Storage**: single-file SQLite index (embeddings stored as float32
  BLOBs alongside snippet metadata), content-addressable snippet IDs
- **Search**: FAISS flat index, cosine similarity via L2-normalized
  inner product, exact (no approximation)
- **Incremental indexing**: `pysnippet update` re-embeds only new or
  changed files, using a two-tier mtime-then-content-hash check so
  unchanged files are never even read
- **Config**: `.pysnippetrc` (TOML) for default model, `top_k`, ignore
  patterns, and language filters, discovered the same way the index
  itself is (walking upward from the current directory)
- **Display**: syntax-highlighted snippet rendering via `rich`/`pygments`

302 tests, 99% coverage. CI on Python 3.9–3.12.
