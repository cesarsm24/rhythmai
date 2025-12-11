"""
Script para poblar la base de datos vectorial con playlists de Deezer.

Permite configurar múltiples playlists con sus estados emocionales asociados,
extrayendo canciones y generando embeddings de texto para búsqueda semántica.
"""

import sys
import re
from tqdm import tqdm
from collections import Counter

from rhythmai.core.embeddings import EmbeddingModel
from rhythmai.stores.factory import get_vector_store
from rhythmai.core.deezer_client import DeezerClient
from rhythmai.config import Config


CUSTOM_PLAYLISTS = [
    {
        "playlist_id": "1910358422",
        "mood": "música triste y melancólica para reflexionar y sentir tristeza",
        "genre": "sad",
        "limit": 20
    },
    {
        "playlist_id": "8605647702",
        "mood": "música triste y emotiva para procesar dolor y pérdida",
        "genre": "sad",
        "limit": 20
    },
    {
        "playlist_id": "4695650924",
        "mood": "música melancólica para momentos de decepción y desilusión",
        "genre": "sad",
        "limit": 20
    },
    {
        "playlist_id": "9565720722",
        "mood": "música tranquila y relajante para calmar ansiedad y miedo",
        "genre": "chill",
        "limit": 20
    },
    {
        "playlist_id": "7421024704",
        "mood": "música alegre y feliz para sentirse bien y bailar",
        "genre": "happy",
        "limit": 20
    },
    {
        "playlist_id": "1613950585",
        "mood": "música emocionante y energética para fiestas y celebrar",
        "genre": "party",
        "limit": 20
    },
    {
        "playlist_id": "14053981301",
        "mood": "música optimista y positiva para sentirse motivado",
        "genre": "pop",
        "limit": 20
    },
    {
        "playlist_id": "10064138882",
        "mood": "música divertida y animada para pasarlo bien",
        "genre": "party",
        "limit": 20
    },
    {
        "playlist_id": "1605532135",
        "mood": "música romántica y amorosa para sentir amor y pasión",
        "genre": "pop",
        "limit": 20
    },
    {
        "playlist_id": "925133345",
        "mood": "música energética y motivadora para entrenar en el gimnasio",
        "genre": "workout",
        "limit": 20
    },
    {
        "playlist_id": "6808852144",
        "mood": "música bailable y animada para bailar y moverse",
        "genre": "dance",
        "limit": 20
    },
    {
        "playlist_id": "1290316405",
        "mood": "música relajante y tranquila para descansar y desconectar",
        "genre": "chill",
        "limit": 20
    },
    {
        "playlist_id": "733113466",
        "mood": "música suave y relajante para dormir y descansar profundamente",
        "genre": "chill",
        "limit": 20
    },
    {
        "playlist_id": "13983981261",
        "mood": "música intensa y agresiva para liberar enojo y frustración",
        "genre": "rock",
        "limit": 20
    },
    {
        "playlist_id": "230734183",
        "mood": "música alternativa y rockera para canalizar molestia",
        "genre": "rock",
        "limit": 20
    },
    {
        "playlist_id": "1282483245",
        "mood": "música popular y actual para cualquier momento",
        "genre": "pop",
        "limit": 20
    }
]


def extract_playlist_id(playlist_input):
    """
    Extrae el identificador de playlist desde una URL o valida un ID directo.

    Args:
        playlist_input (str): URL de Deezer o identificador numérico.

    Returns:
        str: Identificador de la playlist extraído.

    Raises:
        ValueError: Si no se puede extraer un ID válido.

    Examples:
        >>> extract_playlist_id("https://www.deezer.com/es/playlist/1910358422")
        '1910358422'
        >>> extract_playlist_id("1910358422")
        '1910358422'
    """
    if 'deezer.com' in playlist_input:
        match = re.search(r'/playlist/(\d+)', playlist_input)
        if match:
            return match.group(1)

    if playlist_input.isdigit():
        return playlist_input

    raise ValueError(f"No se pudo extraer el ID de playlist de: {playlist_input}")


print("=" * 70)
print("RHYTHMAI - POBLACIÓN DE BASE DE DATOS CON PLAYLISTS")
print("=" * 70)

print("\nInicializando componentes...")

# Inicializar modelo de embeddings
try:
    embedder = EmbeddingModel()
    print("EmbeddingModel inicializado correctamente")
except Exception as e:
    print(f"Error al inicializar EmbeddingModel: {e}")
    sys.exit(1)

# Inicializar vector store
try:
    db = get_vector_store()
    print("Vector store inicializado correctamente")
    print(f"Canciones actuales en base de datos: {db.count()}")
except Exception as e:
    print(f"Error al inicializar vector store: {e}")
    sys.exit(1)

# Inicializar cliente de Deezer
try:
    deezer = DeezerClient()
    print("DeezerClient inicializado correctamente")
except Exception as e:
    print(f"Error al inicializar DeezerClient: {e}")
    sys.exit(1)

songs = []
text_descriptions = []
seen_ids = set()

print(f"\n{'='*70}")
print(f"Procesando {len(CUSTOM_PLAYLISTS)} playlists configuradas...")
print(f"{'='*70}")

# Procesar cada playlist configurada
for idx, playlist_config in enumerate(CUSTOM_PLAYLISTS, 1):
    try:
        playlist_id = playlist_config['playlist_id']
        mood = playlist_config['mood']
        genre = playlist_config['genre']
        limit = playlist_config.get('limit', 20)

        print(f"\n[{idx}/{len(CUSTOM_PLAYLISTS)}] Procesando playlist ID: {playlist_id}")
        print(f"    Estado emocional: {mood}")
        print(f"    Género: {genre}")

        tracks = deezer.get_playlist_tracks(playlist_id, limit=limit)

        if not tracks:
            print(f"  No se pudieron obtener tracks de la playlist {playlist_id}")
            continue

        print(f"  Tracks encontrados: {len(tracks)}")

        count = 0
        duplicates = 0

        # Procesar cada track de la playlist
        for track in tqdm(tracks, desc=f"  {genre}", leave=False):
            try:
                if not track or not isinstance(track, dict):
                    continue

                song_id = track.get('id')
                if not song_id:
                    continue

                # Omitir duplicados
                if song_id in seen_ids:
                    duplicates += 1
                    continue

                seen_ids.add(song_id)

                # Extraer información del track
                song_name = track.get('name', 'Unknown')
                artist_name = track.get('artist', 'Unknown')
                song_url = track.get('url', '#')
                preview_url = track.get('preview_url')
                album_image = track.get('album_image')

                description = mood

                song = {
                    'id': str(song_id),
                    'name': song_name,
                    'artist': artist_name,
                    'description': description,
                    'genre': genre,
                    'url': song_url,
                    'album_image': album_image,
                    'preview_url': preview_url
                }

                songs.append(song)
                text_descriptions.append(description)

                count += 1

            except Exception as e:
                print(f"\n  Error procesando track: {e}")
                continue

        if duplicates > 0:
            print(f"  {count} canciones procesadas ({duplicates} duplicados omitidos)")
        else:
            print(f"  {count} canciones procesadas")

    except Exception as e:
        print(f"  Error procesando playlist '{playlist_id}': {str(e)[:100]}")
        continue

# Validar que se obtuvieron canciones
if len(songs) == 0:
    print("\nERROR: No se pudieron obtener canciones de ninguna playlist.")
    print("Verifica que:")
    print("  - Los IDs de playlists sean correctos")
    print("  - Tengas conexión a internet")
    print("  - Las playlists sean públicas en Deezer")
    sys.exit(1)

print(f"\n{'='*70}")
print(f"Total canciones recolectadas: {len(songs)}")
print(f"Generando embeddings para {len(songs)} canciones...")
print(f"{'='*70}")

# Generar embeddings
print("\nVectorizando descripciones de texto...")
try:
    final_embeddings = embedder.model.encode(
        text_descriptions,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    print(f"Embeddings generados: {final_embeddings.shape}")
except Exception as e:
    print(f"Error vectorizando texto: {e}")
    sys.exit(1)

# Validar dimensionalidad de embeddings
expected_dims = 384
if final_embeddings.shape[1] != expected_dims:
    print(f"\nERROR: Dimensionalidad incorrecta.")
    print(f"   Esperado: {expected_dims} dimensiones")
    print(f"   Obtenido: {final_embeddings.shape[1]} dimensiones")
    sys.exit(1)

# Guardar en base de datos vectorial
print(f"\n{'='*70}")
print("Guardando en base de datos vectorial...")
try:
    db.add_songs(songs, final_embeddings)
    print("Canciones guardadas exitosamente")
except Exception as e:
    print(f"Error guardando en la base de datos: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Mostrar resumen final
print(f"\n{'=' * 70}")
print("COMPLETADO")
print(f"{'=' * 70}")
print(f"Total canciones en base de datos: {db.count()}")
print(f"Géneros disponibles: {', '.join(db.get_all_genres())}")
print(f"Dimensión de embeddings: {final_embeddings.shape[1]}")

stats = db.get_stats()
print(f"\nEstadísticas por género:")

# Mostrar estadísticas según el tipo de vector store
if Config.VECTOR_STORE == "chroma":
    # ChromaDB: usar API de colección
    for genre in stats['genres']:
        try:
            genre_songs = db.collection.get(where={"genre": genre})
            print(f"  {genre}: {len(genre_songs['ids'])} canciones")
        except Exception as e:
            print(f"  {genre}: error contando ({e})")

elif Config.VECTOR_STORE == "faiss":
    # FAISS: contar manualmente desde las canciones procesadas
    genre_counts = Counter(song['genre'] for song in songs)

    for genre in sorted(genre_counts.keys()):
        count = genre_counts[genre]
        print(f"  {genre}: {count} canciones")

else:
    print("  (Estadísticas por género no disponibles para este vector store)")

print(f"\nBase de datos poblada exitosamente")
print("Ahora puedes ejecutar: streamlit run app.py")