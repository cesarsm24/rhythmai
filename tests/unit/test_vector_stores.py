"""
Tests unitarios para las implementaciones de vector store.
"""

import numpy as np
import pytest

from rhythmai.stores.factory import get_vector_store


class TestVectorStore:
    """
    Tests para las implementaciones de vector store.
    """

    @pytest.fixture
    def vector_store(self, temp_db_path):
        """
        Crea una instancia de vector store para testing.

        Args:
            temp_db_path: Ruta temporal para la base de datos de pruebas.

        Returns:
            BaseVectorStore: Instancia de vector store configurada para pruebas.
        """
        import os
        os.environ['VECTOR_STORE'] = 'chroma'
        os.environ['CHROMA_DB_PATH'] = temp_db_path
        return get_vector_store()

    def test_initialization(self, vector_store):
        """
        Verifica que el vector store se inicialice correctamente.
        """
        assert vector_store is not None

    def test_add_songs(self, vector_store, sample_song_data, mock_embedding):
        """
        Verifica la adición de canciones al vector store.
        """
        # Limpiar primero para asegurar estado conocido
        vector_store.clear_all()

        songs = [sample_song_data]
        embeddings = np.array([mock_embedding])

        vector_store.add_songs(songs, embeddings)
        count = vector_store.count()

        assert count == 1

    def test_search(self, vector_store, sample_song_data, mock_embedding):
        """
        Verifica la búsqueda en el vector store.
        """
        # Añadir una canción primero
        songs = [sample_song_data]
        embeddings = np.array([mock_embedding])
        vector_store.add_songs(songs, embeddings)

        # Buscar con embedding similar
        query_embedding = mock_embedding + np.random.randn(384) * 0.01
        results = vector_store.search(query_embedding, n_results=1)

        assert len(results) > 0
        assert 'name' in results[0]
        assert 'artist' in results[0]

    def test_count(self, vector_store, sample_song_data, mock_embedding):
        """
        Verifica el conteo de canciones en el vector store.
        """
        initial_count = vector_store.count()

        songs = [sample_song_data]
        embeddings = np.array([mock_embedding])
        vector_store.add_songs(songs, embeddings)

        new_count = vector_store.count()
        assert new_count == initial_count + 1

    def test_clear_all(self, vector_store, sample_song_data, mock_embedding):
        """
        Verifica la limpieza completa del vector store.
        """
        # Añadir canciones
        songs = [sample_song_data]
        embeddings = np.array([mock_embedding])
        vector_store.add_songs(songs, embeddings)

        # Limpiar
        vector_store.clear_all()

        assert vector_store.count() == 0

    def test_search_empty_store(self, vector_store, mock_embedding):
        """
        Verifica la búsqueda en un vector store vacío.
        """
        results = vector_store.search(mock_embedding, n_results=5)
        assert len(results) == 0

    def test_multiple_songs(self, vector_store, mock_embedding):
        """
        Verifica la adición y búsqueda de múltiples canciones.
        """
        # Limpiar primero
        vector_store.clear_all()

        songs = [
            {
                "id": f"song_{i}",
                "name": f"Test Song {i}",
                "artist": f"Artist {i}",
                "genre": "pop",
                "description": f"Description {i}",
                "url": f"https://example.com/song{i}"
            }
            for i in range(5)
        ]
        # Asegurar que los embeddings sean float32 para compatibilidad con FAISS
        embeddings = np.array(
            [mock_embedding + np.random.randn(384) * 0.1 for _ in range(5)],
            dtype=np.float32
        )

        vector_store.add_songs(songs, embeddings)

        assert vector_store.count() == 5

        results = vector_store.search(mock_embedding, n_results=3)
        assert len(results) == 3