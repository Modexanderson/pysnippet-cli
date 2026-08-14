# pysnippet-cli

Local, GPU-accelerated code snippet search using embeddings — find code by meaning, not just text.

[![CI](https://github.com/modexanderson/pysnippet-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/modexanderson/pysnippet-cli/actions)
[![PyPI](https://img.shields.io/pypi/v/pysnippet-cli.svg)](https://pypi.org/project/pysnippet-cli/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## What it does

`pysnippet` indexes your codebase into a local vector store of code snippets
(functions, classes, blocks) and lets you search it with natural-language
queries — no server, no API key, everything runs on your machine.

```bash
pysnippet index ./my-project
pysnippet find "parse a json config file"
pysnippet show <snippet-id>
```

## Installation

```bash
pip install pysnippet-cli
```

Requires Python 3.9+. The first `index` run downloads a small (~80MB)
embedding model from Hugging Face and caches it locally — after that,
everything runs offline.

## Usage

### Index a project

```console
$ pysnippet index .
Indexing . ...
Indexed 128 snippets from 15 files -> /path/to/project/.pysnippet/index.db
```

This walks the directory, splits every source file into logical
snippets (functions, classes, methods — or blank-line-delimited blocks
for languages without a dedicated parser), embeds them locally, and
stores the result in `.pysnippet/index.db` inside the indexed
directory. Add `.pysnippet/` to your `.gitignore`.

### Search

```console
$ pysnippet find "find the matching closing brace for a string"
1. [0.590] function find_matching_brace  (pysnippet_cli/parsers/brace_scanner.py:12)  id=802808840188
    def find_matching_brace(text: str, open_index: int) -> int | None:
        """Given the index of an opening '{' in `text`, return the index of
        its matching closing '}', skipping braces ins ...

2. [0.544] function parse_brace_language  (pysnippet_cli/parsers/regex_parser.py:83)  id=de7eab58514d
    def parse_brace_language(content: str, *, file_path: str, language: str) -> list[Snippet] | None:
        config = _LANGUAGES.get(language)
        if config is None:
```

`find` works from any subdirectory of an indexed project — it searches
upward for the nearest `.pysnippet/index.db`, the same way `git`
locates `.git`.

```bash
pysnippet find "retry a failed http request" --top-k 3
```

### Show a full snippet

```console
$ pysnippet show 802808840188
```

Renders the snippet in a bordered, syntax-highlighted panel (via
`rich`/`pygments`) with line numbers matching its actual position in
the source file.

### Keep the index up to date

```console
$ pysnippet update
Updating index for /path/to/project ...
Added 1, changed 1, removed 0, unchanged 13 (8 snippets re-embedded) -> /path/to/project/.pysnippet/index.db
```

Only files that are new or have actually changed get re-embedded.
Unchanged files aren't even read: a file's mtime is checked first, and
its content is only re-read and re-hashed if the mtime differs — so a
`git checkout` or `touch` that bumps mtimes without changing content
costs a fast read, not a re-embed. Files deleted from disk have their
snippets removed automatically.

## Supported languages

| Language | Extraction |
|---|---|
| Python | AST-based — functions, classes, methods (`Class.method`), decorator-inclusive spans |
| JavaScript / TypeScript | Regex + brace matching — functions, classes, arrow functions, methods |
| Dart | Regex + brace matching — functions, classes, methods |
| Go, Rust, Java, Kotlin, C, C++, C#, Ruby, PHP, Swift | Generic blank-line-delimited blocks |

Every language falls back to generic block splitting if nothing more
specific is available, so nothing is ever skipped — just chunked less
precisely for languages without a dedicated parser yet.

## Configuration

Drop a `.pysnippetrc` (TOML) in your project root to set defaults.
Like the index itself, it's found by searching upward from wherever
you run a command — so it works the same from any subdirectory.

```toml
# .pysnippetrc
model = "all-MiniLM-L6-v2"
top_k = 10
ignore = ["*.generated.py", "vendor/", "tests/fixtures/*"]
languages = ["python", "javascript", "typescript"]
```

| Key | Default | Description |
|---|---|---|
| `model` | `all-MiniLM-L6-v2` | Any [sentence-transformers](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) model name |
| `top_k` | `5` | Default number of `find` results |
| `ignore` | `[]` | Glob patterns; a trailing `/` prunes a whole directory, anything else matches file paths |
| `languages` | all supported | Restrict indexing to specific languages |

CLI flags (`--model`, `--top-k`) always override the config file.

## How it works

```
index/update:  walk directory -> parse into snippets -> embed locally -> store in SQLite
find:          embed query -> FAISS cosine search over stored embeddings -> rank + display
```

- **Walker**: recursively scans a directory, skipping `.git`,
  `node_modules`, `.venv`, build output, and similar noise by default
- **Parsers**: language-aware extraction where available (Python AST;
  JS/TS/Dart via regex + brace matching), generic block splitting
  everywhere else
- **Embeddings**: [sentence-transformers](https://www.sbert.net/),
  entirely local — no API calls. GPU is used automatically for
  embedding generation if `torch` detects CUDA; the search step itself
  runs on CPU via `faiss-cpu`, which is fast enough at single-project
  scale and installs far more reliably than `faiss-gpu` across
  platforms
- **Storage**: a single SQLite file per indexed project — embeddings
  are stored as float32 BLOBs alongside snippet metadata, so the index
  can't get out of sync with itself
- **Search**: a FAISS flat (exact, brute-force) index over
  L2-normalized vectors, equivalent to cosine similarity, with no
  approximation error

## Development

```bash
git clone https://github.com/modexanderson/pysnippet-cli.git
cd pysnippet-cli
pip install -e ".[dev]"
pytest
ruff check .
```

302 tests, 99% coverage. CI runs on Python 3.9–3.12.

## License

MIT License — see [LICENSE](LICENSE) for details.
