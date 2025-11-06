import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embeddings import EmbeddingModel
from src.core.vector_db import VectorDB
from src.core.spotify_client import SpotifyClient

print("🌱 Poblando base de datos con canciones...")

embedder = EmbeddingModel()
db = VectorDB()
spotify = SpotifyClient()

# Canciones de ejemplo con descripciones
songs_data = [
    {"query": "happy upbeat dance", "description": "música alegre y energética para bailar", "genre": "dance"},
    {"query": "sad acoustic", "description": "canciones tristes y melancólicas", "genre": "acoustic"},
    {"query": "workout energy", "description": "música energética para hacer ejercicio", "genre": "electronic"},
    {"query": "study focus", "description": "música tranquila para concentrarse y estudiar", "genre": "ambient"},
    {"query": "romantic love", "description": "canciones románticas y emotivas", "genre": "r-n-b"},
    {"query": "chill relax", "description": "música relajante y tranquila", "genre": "indie"},
    {"query": "party night", "description": "música para fiestas y celebrar", "genre": "pop"},
    {"query": "angry rock", "description": "rock intenso y enérgico", "genre": "rock"},
]

songs = []
descriptions = []

for item in songs_data:
    tracks = spotify.search_track(item["query"], limit=3)

    for track in tracks:
        song = {
            'id': track['uri'].split(':')[-1],
            'name': track['name'],
            'artist': track['artist'],
            'description': item['description'],
            'genre': item['genre'],
            'url': track['url']
        }
        songs.append(song)
        descriptions.append(item['description'])
        print(f"  ✅ {track['name']} - {track['artist']}")

# Generar embeddings
print("\n🧠 Generando embeddings...")
embeddings = embedder.encode_batch(descriptions)

# Guardar en base de datos
print("💾 Guardando en base de datos...")
db.add_songs(songs, embeddings)

print(f"\n✅ ¡Listo! {len(songs)} canciones añadidas")
print(f"📊 Total en base de datos: {db.count()}")