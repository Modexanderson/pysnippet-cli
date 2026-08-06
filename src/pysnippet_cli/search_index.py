"""FAISS-based similarity search over stored embeddings.

Uses a flat (exact, brute-force) index over L2-normalized vectors with
inner-product search, which is equivalent to cosine similarity. Flat
indices have no approximation error and are fast enough for the
snippet counts a single project produces -- thousands to low tens of
thousands of vectors search in low-single-digit milliseconds on CPU.
A larger corpus could swap in an IVF/HNSW index without changing the
public API here.

GPU acceleration in this tool comes primarily from embedding
generation (sentence-transformers/torch automatically uses CUDA when
available); the search step itself runs on CPU via faiss-cpu, which is
plenty fast at this scale and installs reliably across platforms
(faiss-gpu's pip distribution is far less consistent, especially on
Windows).
"""

from __future__ import annotations

import numpy as np


class SearchIndex:
    """An in-memory FAISS index mapping vector position to snippet id."""

    def __init__(self, ids: list[str], embeddings: np.ndarray) -> None:
        if len(ids) != len(embeddings):
            raise ValueError(
                f"ids ({len(ids)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        self._ids = list(ids)

        if embeddings.size == 0 or not ids:
            self._index = None
            return

        import faiss

        normalized = _normalize(embeddings)
        index = faiss.IndexFlatIP(normalized.shape[1])
        index.add(normalized)
        self._index = index

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Return up to `top_k` (snippet_id, similarity) pairs sorted by
        similarity descending. similarity is cosine similarity."""
        if self._index is None or top_k <= 0:
            return []

        effective_k = min(top_k, len(self._ids))
        query = _normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        scores, indices = self._index.search(query, effective_k)

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._ids[idx], float(score)))
        return results


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row so inner product search == cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for zero vectors
    return vectors / norms
