"""Command-line interface for pysnippet-cli."""

from __future__ import annotations

from pathlib import Path

import click

from pysnippet_cli import __version__
from pysnippet_cli.embedding import DEFAULT_MODEL_NAME


@click.group()
@click.version_option(__version__, prog_name="pysnippet")
def main() -> None:
    """pysnippet - find code by meaning, not just text.

    Index a project's source files into a local embedding store, then
    search it with natural-language or code-like queries.
    """


@main.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    default=".",
)
@click.option(
    "--model", default=DEFAULT_MODEL_NAME, show_default=True, help="Embedding model to use."
)
def index(directory: str, model: str) -> None:
    """Index DIRECTORY, building a local searchable snippet store."""
    from pysnippet_cli.embedding import EmbeddingModel
    from pysnippet_cli.indexer import build_index

    click.echo(f"Indexing {directory} ...")
    result = build_index(Path(directory), embedder=EmbeddingModel(model_name=model))

    if result.snippets_indexed == 0:
        click.echo(f"No indexable source files found under {directory}")
        raise SystemExit(1)

    click.echo(
        f"Indexed {result.snippets_indexed} snippets from {result.files_scanned} files "
        f"-> {result.db_path}"
    )


@main.command()
@click.argument("query")
@click.option("-k", "--top-k", default=5, show_default=True, help="Number of results to return.")
def find(query: str, top_k: int) -> None:
    """Search the index for snippets matching QUERY."""
    click.echo(f"Search not yet implemented (query: {query!r}, top_k: {top_k})")
    raise SystemExit(1)


@main.command()
@click.argument("snippet_id")
def show(snippet_id: str) -> None:
    """Display the full snippet identified by SNIPPET_ID."""
    click.echo(f"Show not yet implemented (id: {snippet_id})")
    raise SystemExit(1)


@main.command()
def update() -> None:
    """Incrementally re-index files that changed since the last index."""
    click.echo("Update not yet implemented")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
