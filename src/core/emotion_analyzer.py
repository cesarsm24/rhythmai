from transformers import pipeline
import numpy as np
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config


class EmotionAnalyzer:
    """
    Analizador de emociones con 28 categorías usando RoBERTa GoEmotions
    Proporciona análisis emocional profundo para recomendaciones musicales
    """

    def __init__(self):
        try:
            # Modelo más completo: 28 emociones
            self.classifier = pipeline(
                "text-classification",
                model=Config.EMOTION_MODEL,
                top_k=None,
                device=-1  # CPU (-1), para GPU usar 0
            )

            print(f"✅ Modelo de análisis emocional cargado: {Config.EMOTION_MODEL}")

            # Definir categorías emocionales para música
            self.emotion_categories = {
                'high_energy_positive': ['excitement', 'amusement', 'joy', 'surprise', 'pride'],
                'low_energy_positive': ['gratitude', 'approval', 'admiration', 'love', 'caring', 'optimism', 'relief'],
                'high_energy_negative': ['anger', 'annoyance', 'nervousness', 'fear'],
                'low_energy_negative': ['sadness', 'grief', 'disappointment', 'embarrassment', 'remorse'],
                'neutral': ['neutral', 'realization', 'confusion', 'curiosity'],
                'ambiguous': ['desire', 'disapproval', 'disgust']
            }

        except Exception as e:
            print(f"❌ Error al cargar modelo de análisis emocional: {e}")
            raise

    def analyze(self, text):
        """
        Análisis emocional ultra-detallado con 28 emociones
        Retorna múltiples métricas para recomendación musical precisa

        Args:
            text (str): Texto a analizar

        Returns:
            dict: Análisis completo con emociones, dimensiones, parámetros Spotify, etc.
        """
        try:
            if not text or not isinstance(text, str) or len(text.strip()) < 3:
                raise ValueError("El texto debe tener al menos 3 caracteres")

            # Obtener todas las emociones con scores
            results = self.classifier(text[:512])[0]  # Limitar a 512 tokens
            emotions = {item['label']: item['score'] for item in results}

            # Top 5 emociones dominantes
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]

            # Calcular dimensiones emocionales
            dimensions = self._calculate_emotional_dimensions(emotions)

            # Calcular parámetros de Spotify optimizados
            spotify_params = self._calculate_spotify_params(emotions, dimensions)

            # Obtener géneros sugeridos
            suggested_genres = self._get_suggested_genres(emotions, dimensions)

            # Detectar contexto de la conversación
            context = self._detect_context(emotions, text)

            # Generar resumen interpretable
            summary = self._generate_summary(emotions, top_emotions, dimensions)

            return {
                'emotions': emotions,
                'top_emotions': top_emotions,
                'dominant_emotion': top_emotions[0][0],
                'dominant_score': top_emotions[0][1],

                # Dimensiones emocionales
                'dimensions': dimensions,

                # Parámetros para Spotify
                'spotify_params': spotify_params,

                # Géneros sugeridos
                'suggested_genres': suggested_genres,

                # Contexto conversacional
                'context': context,

                # Resumen interpretable
                'summary': summary
            }

        except Exception as e:
            print(f"❌ Error en análisis emocional: {e}")
            raise

    def _calculate_emotional_dimensions(self, emotions):
        """
        Calcula 5 dimensiones emocionales clave
        """
        # 1. VALENCIA (positivo vs negativo)
        positive = sum([
            emotions.get('joy', 0) * 1.0,
            emotions.get('amusement', 0) * 0.95,
            emotions.get('excitement', 0) * 0.9,
            emotions.get('gratitude', 0) * 0.85,
            emotions.get('love', 0) * 0.9,
            emotions.get('optimism', 0) * 0.8,
            emotions.get('admiration', 0) * 0.75,
            emotions.get('approval', 0) * 0.7,
            emotions.get('caring', 0) * 0.7,
            emotions.get('pride', 0) * 0.8,
            emotions.get('relief', 0) * 0.75,
        ])

        negative = sum([
            emotions.get('sadness', 0) * 1.0,
            emotions.get('grief', 0) * 1.0,
            emotions.get('anger', 0) * 0.95,
            emotions.get('annoyance', 0) * 0.85,
            emotions.get('disappointment', 0) * 0.9,
            emotions.get('fear', 0) * 0.85,
            emotions.get('nervousness', 0) * 0.8,
            emotions.get('embarrassment', 0) * 0.75,
            emotions.get('remorse', 0) * 0.8,
            emotions.get('disgust', 0) * 0.9,
        ])

        valence = max(0, min(1, (positive - negative + 1) / 2))

        # 2. ENERGÍA (activación alta vs baja)
        high_energy = sum([
            emotions.get('excitement', 0) * 1.0,
            emotions.get('anger', 0) * 0.95,
            emotions.get('surprise', 0) * 0.9,
            emotions.get('nervousness', 0) * 0.85,
            emotions.get('fear', 0) * 0.8,
            emotions.get('amusement', 0) * 0.75,
            emotions.get('joy', 0) * 0.7,
        ])

        low_energy = sum([
            emotions.get('sadness', 0) * 1.0,
            emotions.get('grief', 0) * 0.95,
            emotions.get('disappointment', 0) * 0.8,
            emotions.get('embarrassment', 0) * 0.75,
            emotions.get('neutral', 0) * 0.9,
            emotions.get('relief', 0) * 0.7,
        ])

        energy = max(0, min(1, (high_energy - low_energy + 1) / 2))

        # 3. INTENSIDAD (qué tan fuertes son las emociones)
        intensity = np.mean([score for score in emotions.values()])

        # 4. COMPLEJIDAD EMOCIONAL (mezcla de emociones)
        num_significant = sum(1 for score in emotions.values() if score > 0.1)
        complexity = min(1.0, num_significant / 10)

        # 5. ESTABILIDAD (coherencia emocional)
        sorted_scores = sorted(emotions.values(), reverse=True)
        if len(sorted_scores) > 1:
            stability = sorted_scores[0] - sorted_scores[1]
        else:
            stability = sorted_scores[0]

        return {
            'valence': round(valence, 3),
            'energy': round(energy, 3),
            'intensity': round(intensity, 3),
            'complexity': round(complexity, 3),
            'stability': round(stability, 3)
        }

    def _calculate_spotify_params(self, emotions, dimensions):
        """
        Convierte análisis emocional en parámetros de Spotify API
        """
        # Energía base de dimensiones
        energy = dimensions['energy']

        # Ajustar energía según emociones específicas
        if emotions.get('excitement', 0) > 0.5:
            energy = min(1.0, energy + 0.15)
        if emotions.get('sadness', 0) > 0.5:
            energy = max(0.0, energy - 0.2)

        # Valencia (positivo/negativo)
        valence = dimensions['valence']

        # Danceability (ganas de bailar/moverse)
        danceability = (
                emotions.get('joy', 0) * 0.9 +
                emotions.get('excitement', 0) * 1.0 +
                emotions.get('amusement', 0) * 0.85 +
                emotions.get('optimism', 0) * 0.7
        )
        danceability = max(0, min(1, danceability))

        # Acousticness (acústico vs electrónico)
        acousticness = (
                emotions.get('sadness', 0) * 0.8 +
                emotions.get('love', 0) * 0.7 +
                emotions.get('caring', 0) * 0.6 +
                emotions.get('gratitude', 0) * 0.7 +
                emotions.get('neutral', 0) * 0.5
        )
        acousticness = max(0, min(1, acousticness))

        # Instrumentalness (¿cuánta voz?)
        instrumentalness = dimensions['complexity'] * 0.6

        # Loudness/Intensidad
        loudness = (
                emotions.get('anger', 0) * 1.0 +
                emotions.get('excitement', 0) * 0.9 +
                emotions.get('surprise', 0) * 0.7
        )
        loudness = max(0, min(1, loudness))

        # Tempo (BPM estimado)
        if energy > 0.7:
            tempo_range = (120, 140)
        elif energy > 0.4:
            tempo_range = (90, 120)
        else:
            tempo_range = (60, 90)

        return {
            'target_energy': round(energy, 2),
            'target_valence': round(valence, 2),
            'target_danceability': round(danceability, 2),
            'target_acousticness': round(acousticness, 2),
            'target_instrumentalness': round(instrumentalness, 2),
            'target_loudness': round(loudness, 2),
            'tempo_range': tempo_range
        }

    def _get_suggested_genres(self, emotions, dimensions):
        """
        Sugiere géneros musicales basados en emociones y dimensiones
        """
        genres = []

        # Mapeo emocional detallado a géneros
        emotion_genre_map = {
            'joy': ['pop', 'happy', 'dance', 'party'],
            'excitement': ['electronic', 'dance', 'party', 'house'],
            'amusement': ['indie', 'alternative', 'funk', 'disco'],
            'sadness': ['sad', 'acoustic', 'indie', 'folk'],
            'grief': ['ambient', 'classical', 'sad', 'folk'],
            'anger': ['rock', 'metal', 'punk', 'hard-rock'],
            'annoyance': ['alternative', 'rock', 'indie', 'grunge'],
            'love': ['r-n-b', 'soul', 'jazz', 'blues'],
            'caring': ['acoustic', 'folk', 'indie', 'singer-songwriter'],
            'fear': ['ambient', 'electronic', 'classical', 'instrumental'],
            'nervousness': ['indie', 'alternative', 'electronic', 'jazz'],
            'surprise': ['indie', 'pop', 'alternative', 'electronic'],
            'optimism': ['indie', 'pop', 'alternative', 'happy'],
            'gratitude': ['acoustic', 'folk', 'indie', 'jazz'],
            'admiration': ['classical', 'jazz', 'blues', 'soul'],
            'disappointment': ['alternative', 'indie', 'acoustic', 'sad'],
            'embarrassment': ['indie', 'acoustic', 'alternative', 'folk'],
            'neutral': ['pop', 'indie', 'alternative', 'acoustic'],  # ← Cambiado
            'confusion': ['alternative', 'electronic', 'indie', 'jazz'],  # ← Cambiado
            'curiosity': ['indie', 'alternative', 'jazz', 'world'],
            'pride': ['pop', 'dance', 'electronic', 'hip-hop'],
            'relief': ['acoustic', 'indie', 'ambient', 'folk'],  # ← Cambiado
            'remorse': ['sad', 'acoustic', 'indie', 'folk'],
            'disgust': ['metal', 'punk', 'rock', 'industrial'],
            'desire': ['r-n-b', 'soul', 'electronic', 'jazz'],
            'disapproval': ['alternative', 'punk', 'rock', 'indie'],
            'realization': ['indie', 'alternative', 'ambient', 'electronic'],
        }

        # Obtener géneros de las top 3 emociones
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, score in sorted_emotions:
            if score > 0.1 and emotion in emotion_genre_map:
                genres.extend(emotion_genre_map[emotion][:2])

        # Ajustar por dimensiones
        if dimensions['energy'] > 0.7:
            genres.extend(['high-energy', 'workout', 'power'])
        elif dimensions['energy'] < 0.3:
            genres.extend(['sleep', 'calm', 'study'])

        if dimensions['valence'] > 0.7:
            genres.extend(['feel-good', 'summer', 'party'])
        elif dimensions['valence'] < 0.3:
            genres.extend(['sad', 'melancholic', 'rainy-day'])

        # Remover duplicados manteniendo orden
        seen = set()
        unique_genres = []
        for g in genres:
            if g not in seen:
                seen.add(g)
                unique_genres.append(g)

        return unique_genres[:5]

    def _detect_context(self, emotions, text):
        """
        Detecta el contexto de la conversación
        """
        text_lower = text.lower()
        contexts = []

        # Detectar actividades
        if any(word in text_lower for word in ['estudiar', 'estudio', 'trabajar', 'trabajo', 'concentrar', 'focus']):
            contexts.append('study/work')
        if any(word in text_lower for word in ['gym', 'correr', 'entrenar', 'ejercicio', 'deporte', 'workout']):
            contexts.append('workout')
        if any(word in text_lower for word in ['dormir', 'relajar', 'descansar', 'calm', 'sleep', 'relax']):
            contexts.append('relax/sleep')
        if any(word in text_lower for word in ['fiesta', 'bailar', 'salir', 'party', 'dance']):
            contexts.append('party')
        if any(word in text_lower for word in ['conducir', 'manejar', 'carretera', 'viaje', 'driving', 'road']):
            contexts.append('driving')
        if any(word in text_lower for word in ['llorar', 'cry', 'triste momento', 'mal día', 'bad day']):
            contexts.append('emotional_release')

        # Detectar momento del día
        if any(word in text_lower for word in ['mañana', 'morning', 'despertar', 'wake']):
            contexts.append('morning')
        elif any(word in text_lower for word in ['noche', 'night', 'tarde noche', 'evening']):
            contexts.append('night')

        return contexts if contexts else ['general']

    def _generate_summary(self, emotions, top_emotions, dimensions):
        """
        Genera un resumen interpretable del estado emocional
        """
        dominant = top_emotions[0][0]
        score = top_emotions[0][1]

        # Describir intensidad
        if score > 0.7:
            intensity_desc = "muy claramente"
        elif score > 0.5:
            intensity_desc = "claramente"
        elif score > 0.3:
            intensity_desc = "moderadamente"
        else:
            intensity_desc = "levemente"

        # Describir complejidad
        if dimensions['complexity'] > 0.6:
            complexity_desc = "Tu estado emocional es complejo, con varias emociones mezcladas."
        elif dimensions['complexity'] > 0.3:
            complexity_desc = "Tienes algunas emociones entremezcladas."
        else:
            complexity_desc = "Tu emoción es bastante clara y definida."

        # Describir energía
        if dimensions['energy'] > 0.7:
            energy_desc = "con mucha energía"
        elif dimensions['energy'] > 0.4:
            energy_desc = "con energía moderada"
        else:
            energy_desc = "con poca energía"

        # Describir valencia
        if dimensions['valence'] > 0.6:
            valence_desc = "de manera positiva"
        elif dimensions['valence'] > 0.4:
            valence_desc = "de manera neutral"
        else:
            valence_desc = "de manera negativa"

        # Generar resumen
        summary = (
            f"Te sientes {intensity_desc} {dominant}, "
            f"{energy_desc} y {valence_desc}. "
            f"{complexity_desc}"
        )

        # Añadir emociones secundarias si son relevantes
        if len(top_emotions) > 1 and top_emotions[1][1] > 0.2:
            secondary = top_emotions[1][0]
            summary += f" También hay indicios de {secondary}."

        return summary


# Testing
if __name__ == "__main__":
    print("🧪 Probando analizador de emociones...")

    analyzer = EmotionAnalyzer()

    test_texts = [
        "Me siento muy feliz y emocionado, hoy es un gran día",
        "Estoy triste y decepcionado, las cosas no salieron como esperaba",
        "Tengo mucha ansiedad por el examen de mañana pero también estoy un poco emocionado",
        "Necesito música para concentrarme mientras estudio"
    ]

    for text in test_texts:
        print(f"\n📝 Analizando: '{text}'")
        result = analyzer.analyze(text)
        print(f"✅ Emoción dominante: {result['dominant_emotion']} ({result['dominant_score']:.2%})")
        print(f"   Energía: {result['dimensions']['energy']:.2f}")
        print(f"   Valencia: {result['dimensions']['valence']:.2f}")
        print(f"   Géneros: {', '.join(result['suggested_genres'][:3])}")
        print(f"   Resumen: {result['summary']}")