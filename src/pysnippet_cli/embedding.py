"""Local embedding generation for code snippets via sentence-transformers.

Runs entirely on-device -- no API calls, no keys. The model is loaded
lazily on first use rather than at import time, since loading a
transformer model takes real time and memory that CLI invocations like
`--help` or `--version` shouldn't have to pay for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from pysnippet_cli.snippet import Snippet

# all-MiniLM-L6-v2 is a small (~80MB), fast, general-purpose sentence
# embedding model. It isn't code-specific, but the name/language prefix
# added in `_embedding_text` gives it useful signal, and it keeps the
# tool usable without a large download or GPU. Configurable per-index
# via `EmbeddingModel(model_name=...)`.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _load_model(model_name: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingModel:
    """Wraps a sentence-transformers model, loaded lazily on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dimension(self) -> int:
        """The output vector size for this model. Loads the model if
        it isn't already loaded."""
        model = self._get_model()
        dim = model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                f"Model {self.model_name!r} did not report an embedding dimension"
            )
        return dim

    def embed_texts(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        """Embed a list of raw strings, returning an (N, dim) float32 array."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_snippets(self, snippets: list[Snippet], *, batch_size: int = 32) -> np.ndarray:
        """Embed a list of Snippets by their content, returning an
        (N, dim) float32 array in the same order as `snippets`."""
        texts = [_embedding_text(s) for s in snippets]
        return self.embed_texts(texts, batch_size=batch_size)


def _embedding_text(snippet: Snippet) -> str:
    """Build the text actually fed to the model for a snippet.

    Prefixing with the language and qualified name gives the model
    useful signal beyond raw code tokens -- e.g. distinguishing a
    Python `parse` function from a JS `parse` function, or letting a
    query match by function name even when the query and the function
    body don't share vocabulary.
    """
    parts = [f"# {snippet.language}"]
    if snippet.name:
        parts.append(f"# {snippet.name}")
    parts.append(snippet.content)
    return "\n".join(parts)
