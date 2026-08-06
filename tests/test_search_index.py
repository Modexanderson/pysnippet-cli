import numpy as np
import pytest

from pysnippet_cli.search_index import SearchIndex


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestConstruction:
    def test_raises_on_length_mismatch(self) -> None:
        ids = ["a", "b"]
        embeddings = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            SearchIndex(ids, embeddings)

    def test_empty_index(self) -> None:
        index = SearchIndex([], np.empty((0, 0), dtype=np.float32))
        assert len(index) == 0

    def test_len_matches_input(self) -> None:
        ids = ["a", "b", "c"]
        embeddings = np.random.rand(3, 8).astype(np.float32)
        index = SearchIndex(ids, embeddings)
        assert len(index) == 3


class TestSearch:
    def test_finds_exact_match(self) -> None:
        ids = ["a", "b", "c"]
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        index = SearchIndex(ids, embeddings)

        results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_orders_by_similarity_descending(self) -> None:
        ids = ["close", "far", "medium"]
        embeddings = np.array(
            [
                [1.0, 0.1, 0.0],
                [1.0, 5.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        index = SearchIndex(ids, embeddings)

        results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)
        result_ids = [r[0] for r in results]
        assert result_ids == ["close", "medium", "far"]

    def test_top_k_limits_results(self) -> None:
        ids = ["a", "b", "c", "d"]
        embeddings = np.random.rand(4, 4).astype(np.float32)
        index = SearchIndex(ids, embeddings)

        results = index.search(np.random.rand(4).astype(np.float32), top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_index_size(self) -> None:
        ids = ["a", "b"]
        embeddings = np.random.rand(2, 4).astype(np.float32)
        index = SearchIndex(ids, embeddings)

        results = index.search(np.random.rand(4).astype(np.float32), top_k=10)
        assert len(results) == 2

    def test_empty_index_returns_no_results(self) -> None:
        index = SearchIndex([], np.empty((0, 0), dtype=np.float32))
        results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=5)
        assert results == []

    def test_zero_top_k_returns_no_results(self) -> None:
        ids = ["a"]
        embeddings = np.random.rand(1, 4).astype(np.float32)
        index = SearchIndex(ids, embeddings)

        results = index.search(np.random.rand(4).astype(np.float32), top_k=0)
        assert results == []

    def test_scores_are_cosine_similarity_range(self) -> None:
        ids = ["a", "b"]
        embeddings = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        index = SearchIndex(ids, embeddings)

        results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
        scores = {snippet_id: score for snippet_id, score in results}
        assert scores["a"] == pytest.approx(1.0, abs=1e-5)
        assert scores["b"] == pytest.approx(-1.0, abs=1e-5)

    def test_unnormalized_vectors_still_work(self) -> None:
        # Vectors of very different magnitudes should still rank by
        # direction, not length, since the index normalizes internally.
        ids = ["small", "large"]
        embeddings = np.array([[1.0, 0.0], [100.0, 0.0]], dtype=np.float32)
        index = SearchIndex(ids, embeddings)

        results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
        scores = dict(results)
        assert scores["small"] == pytest.approx(scores["large"], abs=1e-4)
