from unittest.mock import MagicMock, patch

import numpy as np

from pysnippet_cli.embedding import DEFAULT_MODEL_NAME, EmbeddingModel, _embedding_text
from pysnippet_cli.snippet import Snippet


def _mock_model(dim: int = 4) -> MagicMock:
    model = MagicMock()
    model.get_embedding_dimension.return_value = dim

    def _encode(texts, **kwargs):
        return np.zeros((len(texts), dim), dtype=np.float32)

    model.encode.side_effect = _encode
    return model


class TestEmbeddingTextFormatting:
    def test_includes_language(self) -> None:
        s = Snippet(file_path="a.py", start_line=1, end_line=1, content="x = 1", language="python")
        text = _embedding_text(s)
        assert "# python" in text

    def test_includes_name_when_present(self) -> None:
        s = Snippet(
            file_path="a.py", start_line=1, end_line=1, content="...", name="Foo.bar",
        )
        text = _embedding_text(s)
        assert "# Foo.bar" in text

    def test_omits_name_line_when_absent(self) -> None:
        s = Snippet(file_path="a.py", start_line=1, end_line=1, content="x = 1", name=None)
        text = _embedding_text(s)
        lines = text.splitlines()
        assert len(lines) == 2  # language line + content, no name line

    def test_content_included(self) -> None:
        s = Snippet(file_path="a.py", start_line=1, end_line=1, content="def foo(): pass")
        text = _embedding_text(s)
        assert "def foo(): pass" in text


class TestEmbeddingModel:
    def test_default_model_name(self) -> None:
        embedder = EmbeddingModel()
        assert embedder.model_name == DEFAULT_MODEL_NAME

    def test_custom_model_name(self) -> None:
        embedder = EmbeddingModel(model_name="custom-model")
        assert embedder.model_name == "custom-model"

    def test_not_loaded_on_construction(self) -> None:
        embedder = EmbeddingModel()
        assert embedder.is_loaded is False

    @patch("pysnippet_cli.embedding._load_model")
    def test_model_loaded_lazily_on_embed_texts(self, mock_load) -> None:
        mock_load.return_value = _mock_model()
        embedder = EmbeddingModel()

        mock_load.assert_not_called()
        embedder.embed_texts(["hello"])
        mock_load.assert_called_once_with(DEFAULT_MODEL_NAME)
        assert embedder.is_loaded is True

    @patch("pysnippet_cli.embedding._load_model")
    def test_model_loaded_only_once(self, mock_load) -> None:
        mock_load.return_value = _mock_model()
        embedder = EmbeddingModel()

        embedder.embed_texts(["a"])
        embedder.embed_texts(["b"])
        assert mock_load.call_count == 1

    def test_empty_texts_returns_empty_without_loading_model(self) -> None:
        embedder = EmbeddingModel()
        result = embedder.embed_texts([])
        assert result.shape == (0, 0)
        assert embedder.is_loaded is False

    @patch("pysnippet_cli.embedding._load_model")
    def test_embed_texts_returns_float32(self, mock_load) -> None:
        mock_load.return_value = _mock_model(dim=8)
        embedder = EmbeddingModel()

        result = embedder.embed_texts(["a", "b", "c"])
        assert result.dtype == np.float32
        assert result.shape == (3, 8)

    @patch("pysnippet_cli.embedding._load_model")
    def test_dimension_property(self, mock_load) -> None:
        mock_load.return_value = _mock_model(dim=384)
        embedder = EmbeddingModel()

        assert embedder.dimension == 384

    @patch("pysnippet_cli.embedding._load_model")
    def test_dimension_raises_when_model_reports_none(self, mock_load) -> None:
        model = MagicMock()
        model.get_embedding_dimension.return_value = None
        mock_load.return_value = model
        embedder = EmbeddingModel()

        try:
            _ = embedder.dimension
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert embedder.model_name in str(e)

    @patch("pysnippet_cli.embedding._load_model")
    def test_embed_snippets_uses_formatted_text(self, mock_load) -> None:
        mock_model = _mock_model()
        mock_load.return_value = mock_model
        embedder = EmbeddingModel()

        snippets = [
            Snippet(file_path="a.py", start_line=1, end_line=1, content="x = 1", name="foo"),
            Snippet(file_path="b.py", start_line=1, end_line=1, content="y = 2", name="bar"),
        ]
        embedder.embed_snippets(snippets)

        called_texts = mock_model.encode.call_args[0][0]
        assert "foo" in called_texts[0]
        assert "bar" in called_texts[1]

    @patch("pysnippet_cli.embedding._load_model")
    def test_embed_snippets_empty_list(self, mock_load) -> None:
        mock_load.return_value = _mock_model()
        embedder = EmbeddingModel()

        result = embedder.embed_snippets([])
        assert result.shape == (0, 0)
        assert embedder.is_loaded is False

    @patch("pysnippet_cli.embedding._load_model")
    def test_batch_size_passed_through(self, mock_load) -> None:
        mock_model = _mock_model()
        mock_load.return_value = mock_model
        embedder = EmbeddingModel()

        embedder.embed_texts(["a", "b"], batch_size=16)
        assert mock_model.encode.call_args.kwargs["batch_size"] == 16

    @patch("pysnippet_cli.embedding._load_model")
    def test_output_order_matches_input_order(self, mock_load) -> None:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 2

        def _encode(texts, **kwargs):
            # Encode each text's length so we can verify ordering downstream.
            return np.array([[len(t), 0.0] for t in texts], dtype=np.float32)

        mock_model.encode.side_effect = _encode
        mock_load.return_value = mock_model

        embedder = EmbeddingModel()
        result = embedder.embed_texts(["a", "abc", "ab"])
        assert list(result[:, 0]) == [1.0, 3.0, 2.0]
