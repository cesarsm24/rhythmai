"""
Tests de integración end-to-end para el sistema RhythmAI.
"""

import numpy as np
import pytest

from rhythmai.core.embeddings import EmbeddingModel
from rhythmai.core.emotion_analyzer import EmotionAnalyzer
from rhythmai.stores.factory import get_vector_store


class TestEndToEnd:
    """
    Tests de integración end-to-end del sistema completo.
    """

    @pytest.fixture
    def setup_system(self, temp_db_path):
        """
        Configura el sistema completo para testing.

        Args:
            temp_db_path: Ruta temporal para la base de datos de pruebas.

        Returns:
            dict: Diccionario con componentes del sistema inicializados.
        """
        import os
        os.environ['VECTOR_STORE'] = 'chroma'
        os.environ['CHROMA_DB_PATH'] = temp_db_path

        emotion_analyzer = EmotionAnalyzer()
        embedding_model = EmbeddingModel()
        vector_store = get_vector_store()

        return {
            'emotion_analyzer': emotion_analyzer,
            'embedding_model': embedding_model,
            'vector_store': vector_store
        }

    def test_complete_pipeline(self, setup_system):
        """
        Verifica el pipeline completo de recomendación.

        Prueba el flujo completo desde análisis emocional hasta búsqueda
        de canciones en la base de datos vectorial.
        """
        emotion_analyzer = setup_system['emotion_analyzer']
        embedding_model = setup_system['embedding_model']
        vector_store = setup_system['vector_store']

        # Paso 1: Analizar emoción
        user_input = "I'm feeling happy and energetic"
        emotion_result = emotion_analyzer.analyze(user_input)

        assert 'dominant_emotion' in emotion_result

        # Paso 2: Generar embedding
        embedding = embedding_model.encode(user_input)

        assert embedding.shape == (384,)

        # Paso 3: Añadir canciones de muestra a la base de datos
        sample_songs = [
            {
                "id": "happy_song_1",
                "name": "Happy Song",
                "artist": "Joy Artist",
                "genre": "pop",
                "description": "An upbeat happy song with energetic vibes",
                "url": "https://example.com/happy1"
            },
            {
                "id": "sad_song_1",
                "name": "Sad Song",
                "artist": "Melancholy Artist",
                "genre": "ballad",
                "description": "A slow melancholic song for sad moments",
                "url": "https://example.com/sad1"
            }
        ]

        song_descriptions = [song['description'] for song in sample_songs]
        song_embeddings = embedding_model.encode_batch(song_descriptions)

        vector_store.add_songs(sample_songs, song_embeddings)

        # Paso 4: Buscar canciones similares
        results = vector_store.search(embedding, n_results=2)

        assert len(results) > 0
        assert 'name' in results[0]

    def test_emotion_to_music_mapping(self, setup_system):
        """
        Verifica que las emociones se mapeen correctamente a recomendaciones musicales.
        """
        emotion_analyzer = setup_system['emotion_analyzer']
        embedding_model = setup_system['embedding_model']

        # Probar emoción de felicidad
        happy_text = "I'm extremely happy and joyful!"
        happy_emotion = emotion_analyzer.analyze(happy_text)
        happy_embedding = embedding_model.encode(happy_text)

        assert happy_emotion['dominant_emotion'] in ['joy', 'excitement', 'neutral']
        assert isinstance(happy_embedding, np.ndarray)

    def test_database_persistence(self, setup_system, temp_db_path):
        """
        Verifica que la base de datos persista los datos correctamente.

        Prueba que los datos sobrevivan a un reinicio del vector store.
        """
        import os

        vector_store = setup_system['vector_store']
        embedding_model = setup_system['embedding_model']

        # Añadir canciones
        songs = [{
            "id": "persist_test_1",
            "name": "Persistence Test",
            "artist": "Test Artist",
            "genre": "test",
            "description": "Testing database persistence",
            "url": "https://example.com/test"
        }]

        embeddings = embedding_model.encode_batch([songs[0]['description']])
        vector_store.add_songs(songs, embeddings)

        initial_count = vector_store.count()

        # Crear nueva instancia (simulando reinicio)
        os.environ['VECTOR_STORE'] = 'chroma'
        os.environ['CHROMA_DB_PATH'] = temp_db_path
        new_store = get_vector_store()

        # Debe tener el mismo conteo
        assert new_store.count() == initial_count