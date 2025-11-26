import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()

# --- CONFIGURACIÓN ---

GENRES = ["pop", "rock", "lofi", "chill", "reggaeton", "rap", "electronic"]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# --- CONEXIÓN A SPOTIFY ---

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
    )
)

# --- MODELO DE EMBEDDINGS ---

print("🔁 Cargando modelo de embeddings…")
model = SentenceTransformer(MODEL_NAME)
print("✅ Modelo cargado.")

# --- CHROMADB ---

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="songs",
    metadata={"hnsw:space": "cosine"}
)

# --- FUNCIÓN PARA DESCRIBIR CANCIONES ---

def build_text(song, features):
    artist = song["artists"][0]["name"]
    title = song["name"]
    
    return (
        f"Song: {title} by {artist}. "
        f"Energy: {features['energy']}. "
        f"Valence: {features['valence']}. "
        f"Tempo: {features['tempo']} BPM."
    )

# --- AÑADIR CANCIONES POR GÉNERO ---

def add_genre_songs(genre, limit=25):
    print(f"\n🔍 Descargando canciones para el género: {genre}…")

    results = sp.search(q=f"genre:{genre}", type="track", limit=limit)
    tracks = results["tracks"]["items"]

    if not tracks:
        print(f"⚠️ No se encontraron canciones para: {genre}")
        return

    for song in tracks:
        song_id = song["id"]

        # Obtener audio features individualmente para evitar 403
        try:
            feat = sp.audio_features([song_id])[0]
        except Exception as e:
            print(f"   ⚠️ No se pudieron obtener datos para: {song['name']} ({song_id})")
            continue

        if feat is None:
            print(f"   ⚠️ Sin audio features: {song['name']}")
            continue

        text = build_text(song, feat)
        embedding = model.encode(text).tolist()

        collection.add(
            ids=[song["id"]],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "title": song["name"],
                "artist": song["artists"][0]["name"],
                "url": song["external_urls"]["spotify"],
                "genre": genre,
                "energy": feat["energy"],
                "valence": feat["valence"],
                "tempo": feat["tempo"]
            }]
        )

        print(f"   ✓ Añadida: {song['name']} - {song['artists'][0]['name']}")

    print(f"✔️ Género {genre} completado.")


# --- PROGRAMA PRINCIPAL ---

if __name__ == "__main__":
    print("🎵 Creando la base vectorial…")
    print("-----------------------------------")

    for g in GENRES:
        add_genre_songs(g)

    print("\n✅ Base vectorial creada con éxito.")
