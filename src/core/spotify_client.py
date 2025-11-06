import spotipy
from spotipy.oauth2 import SpotifyOAuth
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config


class SpotifyClient:
    """
    Cliente para interactuar con Spotify API
    Maneja autenticación y búsqueda de música
    """

    def __init__(self):
        try:
            # Configurar autenticación con Spotify
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=Config.SPOTIFY_CLIENT_ID,
                client_secret=Config.SPOTIFY_CLIENT_SECRET,
                redirect_uri=Config.SPOTIFY_REDIRECT_URI,
                scope=Config.SPOTIFY_SCOPE,
                cache_path=".spotify_cache",
                open_browser=True
            ))

            # Verificar conexión
            user = self.sp.current_user()
            print(f"✅ Conectado a Spotify como: {user['display_name']}")

        except Exception as e:
            print(f"❌ Error al conectar con Spotify: {e}")
            print("⚠️ Verifica tus credenciales en el archivo .env")
            raise

    def search_track(self, query, limit=5):
        """
        Busca canciones en Spotify

        Args:
            query (str): Consulta de búsqueda
            limit (int): Número de resultados

        Returns:
            list: Lista de canciones encontradas
        """
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            tracks = []

            for item in results['tracks']['items']:
                tracks.append({
                    'name': item['name'],
                    'artist': item['artists'][0]['name'],
                    'url': item['external_urls']['spotify'],
                    'uri': item['uri'],
                    'preview_url': item.get('preview_url'),
                    'album_image': item['album']['images'][0]['url'] if item['album']['images'] else None
                })

            return tracks

        except Exception as e:
            print(f"❌ Error en búsqueda de Spotify: {e}")
            return []

    def get_recommendations(self, seed_tracks=None, seed_genres=None,
                            target_energy=None, target_valence=None, limit=10):
        """
        Obtiene recomendaciones basadas en parámetros emocionales

        Args:
            seed_tracks (list): Lista de IDs de canciones semilla
            seed_genres (list): Lista de géneros semilla
            target_energy (float): Nivel de energía objetivo (0-1)
            target_valence (float): Nivel de valencia objetivo (0-1)
            limit (int): Número de recomendaciones

        Returns:
            dict: Recomendaciones de Spotify
        """
        try:
            return self.sp.recommendations(
                seed_tracks=seed_tracks,
                seed_genres=seed_genres,
                target_energy=target_energy,
                target_valence=target_valence,
                limit=limit
            )
        except Exception as e:
            print(f"❌ Error obteniendo recomendaciones: {e}")
            return {'tracks': []}

    def get_user_playlists(self, limit=20):
        """
        Obtiene las playlists del usuario
        """
        try:
            playlists = self.sp.current_user_playlists(limit=limit)
            return playlists['items']
        except Exception as e:
            print(f"❌ Error obteniendo playlists: {e}")
            return []


# Testing
if __name__ == "__main__":
    print("🧪 Probando cliente de Spotify...")

    try:
        client = SpotifyClient()

        # Test búsqueda
        results = client.search_track("happy", limit=3)
        print(f"✅ Encontradas {len(results)} canciones")
        for track in results:
            print(f"   • {track['name']} - {track['artist']}")

    except Exception as e:
        print(f"❌ Error: {e}")