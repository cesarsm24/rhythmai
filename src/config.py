import os
from dotenv import load_dotenv
from pathlib import Path

# Cargar variables de entorno desde .env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """
    Configuración centralizada de la aplicación
    """

    # ========================
    # SPOTIFY API
    # ========================
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    SPOTIFY_SCOPE = "user-read-playback-state user-modify-playback-state playlist-read-private user-read-recently-played"

    # ========================
    # MODELOS DE IA
    # ========================
    # Modelo de embeddings (384 dimensiones, ~80 MB)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Modelo de análisis emocional (28 emociones, ~500 MB)
    EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"

    # ========================
    # RUTAS Y DIRECTORIOS
    # ========================
    BASE_DIR = Path(__file__).parent.parent
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
    MEMORY_PATH = os.getenv("MEMORY_PATH", str(BASE_DIR / "memory"))
    DATA_PATH = str(BASE_DIR / "data")

    # ========================
    # CONFIGURACIÓN DE MEMORIA
    # ========================
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", 50))
    MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", 10))

    # ========================
    # VALIDACIÓN
    # ========================
    @classmethod
    def validate(cls):
        """
        Valida que todas las configuraciones necesarias estén presentes
        """
        errors = []

        if not cls.SPOTIFY_CLIENT_ID:
            errors.append("SPOTIFY_CLIENT_ID no está configurado en .env")

        if not cls.SPOTIFY_CLIENT_SECRET:
            errors.append("SPOTIFY_CLIENT_SECRET no está configurado en .env")

        if errors:
            error_msg = "❌ Errores de configuración:\n" + "\n".join(f"  • {e}" for e in errors)
            raise ValueError(error_msg)

        return True

    @classmethod
    def print_config(cls):
        """
        Imprime la configuración actual (sin mostrar secretos)
        """
        print("⚙️ Configuración de RhythmAI:")
        print(f"  • Spotify Client ID: {'✅ Configurado' if cls.SPOTIFY_CLIENT_ID else '❌ No configurado'}")
        print(f"  • Spotify Secret: {'✅ Configurado' if cls.SPOTIFY_CLIENT_SECRET else '❌ No configurado'}")
        print(f"  • Redirect URI: {cls.SPOTIFY_REDIRECT_URI}")
        print(f"  • Modelo de embeddings: {cls.EMBEDDING_MODEL}")
        print(f"  • Modelo emocional: {cls.EMOTION_MODEL}")
        print(f"  • Base de datos: {cls.CHROMA_DB_PATH}")
        print(f"  • Memoria: {cls.MEMORY_PATH}")
        print(f"  • Historial máximo: {cls.MAX_CONVERSATION_HISTORY}")


# Validar al importar
if __name__ != "__main__":
    try:
        Config.validate()
    except ValueError as e:
        print(e)
        print("\n💡 Crea un archivo .env en la raíz del proyecto con tus credenciales de Spotify")