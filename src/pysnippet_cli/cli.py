"""Command-line interface for pysnippet-cli."""

from __future__ import annotations

import click

from pysnippet_cli import __version__


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
def index(directory: str) -> None:
    """Index DIRECTORY, building a local searchable snippet store."""
    click.echo(f"Indexing not yet implemented (target: {directory})")
    raise SystemExit(1)


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
