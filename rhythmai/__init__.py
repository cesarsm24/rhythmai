"""
RhythmAI - Sistema Inteligente de Recomendación Musical.

Sistema de recomendación de música impulsado por IA que combina:
- Análisis emocional mediante modelos transformer
- Búsqueda semántica con bases de datos vectoriales
- Vectorización de texto para matching semántico
- Sistema de memoria y aprendizaje de preferencias del usuario

Autores:
    - César Sánchez Montes
    - Miguel Ángel Campón Iglesias
    - Nicolás Benito Benito

Licencia: MIT
Versión: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "RhythmAI Team"

from rhythmai.config import Config
from rhythmai.core.music_recommender import MusicRecommender
from rhythmai.stores.factory import get_vector_store

__all__ = [
    "Config",
    "MusicRecommender",
    "get_vector_store",
]