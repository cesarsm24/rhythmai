"""
Cliente de Deezer API sin autenticación.

Proporciona acceso simplificado para obtener canciones de playlists públicas
mediante la API de Deezer sin requerir tokens de autenticación.

Todos los metadatos se convierten a tipos primitivos de Python (str, int, float, bool)
para garantizar compatibilidad con sistemas de almacenamiento y serialización.
"""

import requests
import logging

logger = logging.getLogger(__name__)


class DeezerClient:
    """
    Cliente para interactuar con la API pública de Deezer.

    Proporciona método para obtener canciones de playlists públicas,
    que es la funcionalidad principal para poblar la base de datos vectorial.
    """

    BASE_URL = "https://api.deezer.com"

    def __init__(self):
        """
        Inicializa el cliente de Deezer con configuración de sesión HTTP.
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RhythmAI/1.0',
            'Accept': 'application/json'
        })
        logger.info("Cliente Deezer inicializado correctamente")

    def get_playlist_tracks(self, playlist_id, limit=50):
        """
        Obtiene las canciones de una playlist de Deezer.

        Este es el método principal utilizado para poblar la base de datos vectorial.
        Todos los metadatos se extraen como tipos primitivos (str, int, float, bool)
        para garantizar compatibilidad con sistemas de almacenamiento vectorial
        y serialización JSON.

        Args:
            playlist_id (int): Identificador numérico de la playlist.
            limit (int): Número máximo de canciones a obtener.

        Returns:
            list: Lista de canciones con metadatos en tipos primitivos.
                  Cada canción contiene: id, name, artist, url, uri, preview_url, album_image.
        """
        try:
            url = f"{self.BASE_URL}/playlist/{playlist_id}/tracks"
            params = {'limit': limit}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'data' not in data:
                logger.warning(f"No se encontraron datos en playlist {playlist_id}")
                return []

            tracks = []
            for item in data['data']:
                artist_info = item.get('artist', {})
                if isinstance(artist_info, dict):
                    artist_name = artist_info.get('name', 'Unknown')
                else:
                    artist_name = str(artist_info)

                album_info = item.get('album', {})
                if isinstance(album_info, dict):
                    album_image = album_info.get('cover_medium')
                else:
                    album_image = None

                tracks.append({
                    'id': item.get('id'),
                    'name': item.get('title', 'Unknown'),
                    'artist': artist_name,
                    'url': item.get('link', '#'),
                    'uri': f"deezer:track:{item.get('id')}",
                    'preview_url': item.get('preview'),
                    'album_image': album_image
                })

            logger.info(f"Obtenidas {len(tracks)} canciones de playlist {playlist_id}")
            return tracks

        except requests.exceptions.RequestException as e:
            logger.error(f"Error obteniendo playlist {playlist_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error procesando playlist {playlist_id}: {e}")
            return []