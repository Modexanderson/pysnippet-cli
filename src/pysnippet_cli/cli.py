"""Command-line interface for pysnippet-cli."""

from __future__ import annotations

from pathlib import Path

import click

from pysnippet_cli import __version__
from pysnippet_cli.embedding import DEFAULT_MODEL_NAME
from pysnippet_cli.snippet import Snippet


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
    from pysnippet_cli.embedding import EmbeddingModel
    from pysnippet_cli.indexer import find_project_index
    from pysnippet_cli.search_index import SearchIndex
    from pysnippet_cli.store import SnippetStore

    db_path = find_project_index()
    if db_path is None:
        click.echo("No index found. Run `pysnippet index <directory>` first.")
        raise SystemExit(1)

    with SnippetStore(db_path) as store:
        if store.count() == 0:
            click.echo("Index is empty. Run `pysnippet index <directory>` first.")
            raise SystemExit(1)

        model_name = store.get_meta("model_name") or DEFAULT_MODEL_NAME
        embedder = EmbeddingModel(model_name=model_name)

        ids, vectors = store.all_embeddings()
        search_idx = SearchIndex(ids, vectors)

        query_vector = embedder.embed_texts([query])[0]
        results = search_idx.search(query_vector, top_k=top_k)

        if not results:
            click.echo("No matches found.")
            return

        for rank, (snippet_id, score) in enumerate(results, start=1):
            snippet = store.get_snippet(snippet_id)
            if snippet is None:
                continue
            click.echo(_format_result(rank, snippet, score))


def _format_result(rank: int, snippet: Snippet, score: float) -> str:
    label = snippet.kind if snippet.name is None else f"{snippet.kind} {snippet.name}"
    header = f"{rank}. [{score:.3f}] {label}  ({snippet.location()})  id={snippet.id}"
    preview = _preview(snippet.content)
    return f"{header}\n{preview}"


def _preview(content: str, *, max_lines: int = 3, max_chars: int = 200) -> str:
    lines = content.splitlines()[:max_lines]
    text = "\n".join(f"    {line}" for line in lines)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    return text


@main.command()
@click.argument("snippet_id")
def show(snippet_id: str) -> None:
    """Display the full snippet identified by SNIPPET_ID."""
    from pysnippet_cli.display import render_snippet
    from pysnippet_cli.indexer import find_project_index
    from pysnippet_cli.store import SnippetStore

    db_path = find_project_index()
    if db_path is None:
        click.echo("No index found. Run `pysnippet index <directory>` first.")
        raise SystemExit(1)

    with SnippetStore(db_path) as store:
        snippet = store.get_snippet(snippet_id)
        if snippet is None:
            click.echo(f"No snippet found with id {snippet_id!r}")
            raise SystemExit(1)

        render_snippet(snippet)


@main.command()
def update() -> None:
    """Incrementally re-index files that changed since the last index."""
    from pysnippet_cli.indexer import find_project_index, update_index
    from pysnippet_cli.store import SnippetStore

    db_path = find_project_index()
    if db_path is None:
        click.echo("No index found. Run `pysnippet index <directory>` first.")
        raise SystemExit(1)

    with SnippetStore(db_path) as store:
        source_directory = store.get_meta("source_directory")

    if source_directory is None:
        click.echo("Index is missing its source directory. Re-run `pysnippet index`.")
        raise SystemExit(1)

    click.echo(f"Updating index for {source_directory} ...")
    result = update_index(source_directory, db_path=db_path)

    click.echo(
        f"Added {result.files_added}, changed {result.files_changed}, "
        f"removed {result.files_removed}, unchanged {result.files_unchanged} "
        f"({result.snippets_indexed} snippets re-embedded) -> {result.db_path}"
    )


if __name__ == "__main__":
    main()
