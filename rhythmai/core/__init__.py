"""
Módulos principales de RhythmAI.

Contiene los componentes core del sistema de recomendación:
- EmbeddingModel: Vectorización de texto mediante transformers
- EmotionAnalyzer: Análisis emocional con modelos pre-entrenados
- DeezerClient: Cliente de API de Deezer para búsqueda musical
- MusicRecommender: Orquestador principal del sistema de recomendación
"""

from rhythmai.core.embeddings import EmbeddingModel
from rhythmai.core.emotion_analyzer import EmotionAnalyzer
from rhythmai.core.deezer_client import DeezerClient
from rhythmai.core.music_recommender import MusicRecommender

__all__ = [
    "EmbeddingModel",
    "EmotionAnalyzer",
    "DeezerClient",
    "MusicRecommender",
]