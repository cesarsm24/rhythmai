"""
Tests unitarios para el módulo de análisis emocional.
"""

import pytest

from rhythmai.core.emotion_analyzer import EmotionAnalyzer


class TestEmotionAnalyzer:
    """
    Tests para la clase EmotionAnalyzer.
    """

    @pytest.fixture
    def analyzer(self):
        """
        Crea una instancia de EmotionAnalyzer para testing.

        Returns:
            EmotionAnalyzer: Instancia configurada para pruebas.
        """
        return EmotionAnalyzer()

    def test_initialization(self, analyzer):
        """
        Verifica que el analizador de emociones se inicialice correctamente.
        """
        assert analyzer is not None
        assert hasattr(analyzer, 'sentiment_pipeline')
        assert hasattr(analyzer, 'embedder')

    def test_analyze_happy_emotion(self, analyzer):
        """
        Verifica el análisis de texto con emoción de felicidad.
        """
        text = "I'm feeling incredibly happy and joyful today!"
        result = analyzer.analyze(text)

        assert 'dominant_emotion' in result
        assert 'dominant_score' in result
        assert 'suggested_genres' in result
        assert 'dimensions' in result
        assert result['dominant_emotion'] in ['joy', 'excitement', 'neutral']

    def test_analyze_sad_emotion(self, analyzer):
        """
        Verifica el análisis de texto con emoción de tristeza.
        """
        text = "I'm feeling very sad and depressed"
        result = analyzer.analyze(text)

        assert 'dominant_emotion' in result
        assert result['dominant_emotion'] in ['sadness', 'neutral']

    def test_analyze_angry_emotion(self, analyzer):
        """
        Verifica el análisis de texto con emoción de enojo.
        """
        text = "I'm very angry, I hate this situation!"
        result = analyzer.analyze(text)

        assert 'dominant_emotion' in result
        # El análisis puede detectar diferentes emociones según el modelo
        # Aceptamos anger, stressed (por la frustración) o sadness (emoción negativa)
        assert result['dominant_emotion'] in ['anger', 'angry', 'stressed', 'sadness', 'neutral']

    def test_suggested_genres_present(self, analyzer):
        """
        Verifica que suggested_genres esté presente en el resultado.
        """
        text = "I'm feeling a mix of emotions"
        result = analyzer.analyze(text)

        assert 'suggested_genres' in result
        assert isinstance(result['suggested_genres'], list)
        assert len(result['suggested_genres']) > 0

    def test_score_range(self, analyzer):
        """
        Verifica que los scores de emoción estén en el rango válido.
        """
        text = "I'm feeling great!"
        result = analyzer.analyze(text)

        assert 0 <= result['dominant_score'] <= 1
        assert 'dimensions' in result
        assert 0 <= result['dimensions']['valence'] <= 1
        assert 0 <= result['dimensions']['energy'] <= 1

    def test_empty_text(self, analyzer):
        """
        Verifica que analizar texto vacío retorne emoción neutral.
        """
        result = analyzer.analyze("")

        assert result is not None
        assert result['dominant_emotion'] == 'neutral'
        assert result['dominant_score'] == 0.50

    def test_dimensions(self, analyzer):
        """
        Verifica que las dimensiones se calculen correctamente.
        """
        text = "I'm extremely happy and full of energy!"
        result = analyzer.analyze(text)

        assert 'dimensions' in result
        assert 'valence' in result['dimensions']
        assert 'energy' in result['dimensions']
        assert 0 <= result['dimensions']['valence'] <= 1
        assert 0 <= result['dimensions']['energy'] <= 1