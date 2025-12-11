"""
Tests de integración para el sistema de recomendación musical.
"""

import pytest

from rhythmai.core.music_recommender import MusicRecommender


class TestMusicRecommenderIntegration:
    """
    Tests de integración para el sistema MusicRecommender.
    """

    @pytest.fixture
    def recommender(self, temp_db_path):
        """
        Crea una instancia de MusicRecommender para testing.

        Args:
            temp_db_path: Ruta temporal para la base de datos de pruebas.

        Returns:
            MusicRecommender: Instancia configurada para pruebas.
        """
        import os
        os.environ['VECTOR_STORE'] = 'chroma'
        os.environ['CHROMA_DB_PATH'] = temp_db_path
        os.environ['MEMORY_PATH'] = temp_db_path + '/memory'

        return MusicRecommender(user_id="test_user")

    def test_recommend_basic(self, recommender):
        """
        Verifica el flujo básico de recomendación.
        """
        user_input = "I'm feeling happy and energetic"

        result = recommender.recommend(user_input, n_results=5)

        assert 'emotion_analysis' in result
        assert 'vector_results' in result
        assert 'explanation' in result

    def test_emotion_analysis_in_recommendation(self, recommender):
        """
        Verifica que el análisis emocional esté incluido en la recomendación.
        """
        user_input = "I'm feeling sad and need calming music"

        result = recommender.recommend(user_input, n_results=5)

        emotion = result['emotion_analysis']
        assert 'dominant_emotion' in emotion
        assert 'dominant_score' in emotion
        assert 'suggested_genres' in emotion
        assert 'dimensions' in emotion

    def test_different_emotions(self, recommender):
        """
        Verifica recomendaciones para diferentes estados emocionales.
        """
        test_cases = [
            "I'm feeling extremely happy and joyful",
            "I'm sad and need comfort",
            "I'm angry and frustrated",
            "I'm calm and relaxed"
        ]

        for user_input in test_cases:
            result = recommender.recommend(user_input, n_results=3)
            assert 'emotion_analysis' in result
            assert result['emotion_analysis']['dominant_emotion'] is not None

    def test_n_results_parameter(self, recommender):
        """
        Verifica que se respete el parámetro n_results.
        """
        user_input = "I want energetic music"

        result = recommender.recommend(user_input, n_results=3)

        # Puede retornar menos si la base de datos está vacía
        assert len(result.get('vector_results', [])) <= 3

    def test_explanation_generation(self, recommender):
        """
        Verifica que se genere una explicación.
        """
        user_input = "I'm feeling happy"

        result = recommender.recommend(user_input, n_results=5)

        assert 'explanation' in result
        assert isinstance(result['explanation'], str)
        assert len(result['explanation']) > 0

    @pytest.mark.slow
    def test_multiple_recommendations(self, recommender):
        """
        Verifica múltiples recomendaciones consecutivas.
        """
        inputs = [
            "I'm feeling happy",
            "I'm feeling sad",
            "I'm feeling energetic"
        ]

        for user_input in inputs:
            result = recommender.recommend(user_input, n_results=3)
            assert 'emotion_analysis' in result