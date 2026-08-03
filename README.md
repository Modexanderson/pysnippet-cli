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

> **Status:** early scaffold. Indexing and search are not implemented yet —
> see the roadmap below.

## Installation

```bash
pip install pysnippet-cli
```

## Roadmap

- [x] CLI scaffold (`index`, `find`, `show`, `update`)
- [ ] File walker and language-aware snippet splitting (Python, JS/TS, Dart, generic)
- [ ] Local embedding generation (sentence-transformers)
- [ ] GPU-accelerated similarity search (FAISS)
- [ ] Incremental re-indexing
- [ ] Config file support (`.pysnippetrc`)

## Development

```bash
git clone https://github.com/modexanderson/pysnippet-cli.git
cd pysnippet-cli
pip install -e ".[dev]"
pytest
```

## License

MIT License — see [LICENSE](LICENSE) for details.
