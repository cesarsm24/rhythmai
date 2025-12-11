"""
Configuración centralizada de RhythmAI.

Maneja todas las configuraciones de la aplicación, incluyendo
rutas de almacenamiento, modelos de IA y opciones de hardware.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """
    Configuración centralizada de la aplicación RhythmAI.

    Todas las configuraciones se cargan desde variables de entorno
    con valores por defecto apropiados para desarrollo.
    """

    DEEZER_API_URL = "https://api.deezer.com"

    # Modelos de IA (Hugging Face)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMOTION_MODEL = os.getenv("EMOTION_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual")
    EMBEDDING_DIMENSION = 384  # Dimensión de embeddings del modelo all-MiniLM-L6-v2

    BASE_DIR = Path(__file__).parent.parent
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
    FAISS_DB_PATH = os.getenv("FAISS_DB_PATH", str(BASE_DIR / "faiss_db"))
    MEMORY_PATH = os.getenv("MEMORY_PATH", str(BASE_DIR / "memory"))
    DATA_PATH = os.getenv("DATA_PATH", str(BASE_DIR / "data"))

    VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma").lower()

    if VECTOR_STORE not in ["chroma", "faiss"]:
        raise ValueError(f"VECTOR_STORE inválido: {VECTOR_STORE}. Opciones: 'chroma', 'faiss'")

    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", 50))
    MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", 10))

    USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

    RHYTHM_MASTER_KEY = os.getenv("RHYTHM_MASTER_KEY", "default_master_key_change_in_production")

    @classmethod
    def validate(cls):
        """
        Valida que todas las configuraciones necesarias estén presentes.

        Returns:
            bool: True si la configuración es válida.

        Raises:
            ValueError: Si faltan configuraciones requeridas.
        """
        required_paths = [cls.BASE_DIR]

        for path in required_paths:
            if not Path(path).exists():
                Path(path).mkdir(parents=True, exist_ok=True)

        return True

    @classmethod
    def print_config(cls):
        """
        Imprime la configuración actual del sistema sin exponer información sensible.
        """
        print("=" * 60)
        print("Configuración de RhythmAI")
        print("=" * 60)
        print(f"  API de música: Deezer")
        print(f"  Vector Store: {cls.VECTOR_STORE.upper()}")
        print(f"  Modelo de embeddings: {cls.EMBEDDING_MODEL}")
        print(f"  Modelo emocional: {cls.EMOTION_MODEL}")
        print(f"  Dimensión de embeddings: {cls.EMBEDDING_DIMENSION}")
        print(f"  GPU: {'Habilitada' if cls.USE_GPU else 'Deshabilitada (CPU)'}")
        print(f"  Ruta de base de datos: {cls.CHROMA_DB_PATH if cls.VECTOR_STORE == 'chroma' else cls.FAISS_DB_PATH}")
        print(f"  Ruta de memoria: {cls.MEMORY_PATH}")
        print("=" * 60)


if __name__ != "__main__":
    try:
        Config.validate()
    except ValueError as e:
        print(f"Error en configuración: {e}")
        print("Revisa la configuración en el archivo .env")