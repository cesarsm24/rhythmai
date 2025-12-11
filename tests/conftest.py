"""
Configuración de pytest y fixtures compartidos para tests de RhythmAI.
"""

import os
import sys
from pathlib import Path

import pytest

# Añadir raíz del proyecto al path de Python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_text():
    """
    Texto de ejemplo para testing de análisis emocional.

    Returns:
        str: Texto de muestra con emoción positiva.
    """
    return "I'm feeling happy and energetic today!"


@pytest.fixture
def sample_song_data():
    """
    Datos de canción de ejemplo para testing de operaciones de vector store.

    Returns:
        dict: Diccionario con información de canción de prueba.
    """
    return {
        "id": "test_song_1",
        "name": "Test Song",
        "artist": "Test Artist",
        "genre": "pop",
        "description": "An upbeat and energetic pop song",
        "url": "https://example.com/song1"
    }


@pytest.fixture
def mock_embedding():
    """
    Vector de embedding simulado (384 dimensiones).

    Returns:
        np.ndarray: Vector de embedding aleatorio de 384 dimensiones.
    """
    import numpy as np
    return np.random.rand(384).astype('float32')


@pytest.fixture
def temp_db_path(tmp_path):
    """
    Ruta temporal de base de datos para testing.

    Crea un directorio temporal y lo limpia después del test.

    Args:
        tmp_path: Fixture de pytest que proporciona directorio temporal.

    Yields:
        str: Ruta al directorio temporal de base de datos.
    """
    db_path = str(tmp_path / "test_db")
    # Limpiar al finalizar el test
    yield db_path
    import shutil
    if Path(db_path).exists():
        shutil.rmtree(db_path, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_env_vars():
    """
    Resetea variables de entorno antes de cada test.

    Guarda el estado original de las variables de entorno y las restaura
    después de cada test para evitar efectos secundarios entre tests.

    Yields:
        None
    """
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)