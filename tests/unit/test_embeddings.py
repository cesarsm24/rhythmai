"""
Tests unitarios para el módulo de embeddings.
"""

import numpy as np
import pytest

from rhythmai.core.embeddings import EmbeddingModel


class TestEmbeddingModel:
    """
    Tests para la clase EmbeddingModel.
    """

    def test_initialization(self):
        """
        Verifica que el modelo de embeddings se inicialice correctamente.
        """
        model = EmbeddingModel()
        assert model is not None
        assert hasattr(model, 'model')

    def test_encode_single_text(self):
        """
        Verifica la codificación de una cadena de texto individual.
        """
        model = EmbeddingModel()
        text = "I'm feeling happy today"
        embedding = model.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_encode_batch(self):
        """
        Verifica la codificación de un lote de textos.
        """
        model = EmbeddingModel()
        texts = [
            "I'm feeling happy",
            "I'm feeling sad",
            "I'm feeling energetic"
        ]
        embeddings = model.encode_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_encode_empty_string(self):
        """
        Verifica que codificar una cadena vacía lance ValueError.
        """
        model = EmbeddingModel()

        with pytest.raises(ValueError, match="El texto debe ser una cadena no vacía"):
            model.encode("")

    def test_similarity(self):
        """
        Verifica el cálculo de similitud entre embeddings.

        Textos similares deben tener mayor similitud que textos diferentes.
        """
        model = EmbeddingModel()
        text1 = "I'm happy and joyful"
        text2 = "I'm feeling great and excited"
        text3 = "I'm sad and depressed"

        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        emb3 = model.encode(text3)

        # Textos similares deben tener mayor similitud
        similarity_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarity_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

        assert similarity_12 > similarity_13