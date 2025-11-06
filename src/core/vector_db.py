import chromadb
from chromadb.config import Settings
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config


class VectorDB:
    """
    Sistema de base de datos vectorial usando ChromaDB
    Para búsqueda semántica de canciones basada en similitud
    """

    def __init__(self):
        try:
            # Crear cliente persistente de ChromaDB
            self.client = chromadb.PersistentClient(
                path=Config.CHROMA_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Obtener o crear colección de canciones
            self.collection = self.client.get_or_create_collection(
                name="songs",
                metadata={"hnsw:space": "cosine"}  # Usar distancia coseno
            )

            print(f"✅ Base de datos vectorial inicializada: {Config.CHROMA_DB_PATH}")
            print(f"   Canciones en la base de datos: {self.collection.count()}")

        except Exception as e:
            print(f"❌ Error al inicializar base de datos vectorial: {e}")
            raise

    def add_song(self, song_id, name, artist, description, genre, url, embedding):
        """
        Añade una canción individual a la base de datos

        Args:
            song_id (str): ID único de la canción
            name (str): Nombre de la canción
            artist (str): Artista
            description (str): Descripción de la canción/mood
            genre (str): Género musical
            url (str): URL de Spotify
            embedding (list/numpy.ndarray): Vector de embeddings
        """
        try:
            metadata = {
                'name': name,
                'artist': artist,
                'description': description,
                'genre': genre,
                'url': url
            }

            # Convertir numpy array a lista si es necesario
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()

            self.collection.add(
                ids=[str(song_id)],
                embeddings=[embedding],
                metadatas=[metadata]
            )

            print(f"✅ Canción añadida: {name} - {artist}")

        except Exception as e:
            print(f"❌ Error al añadir canción: {e}")
            raise

    def add_songs(self, songs, embeddings):
        """
        Añade múltiples canciones a la base de datos
        """
        try:
            # Verificar que existen datos
            if not songs or len(songs) == 0:
                raise ValueError("Se requieren canciones")

            if embeddings is None or (hasattr(embeddings, 'shape') and embeddings.shape[0] == 0):
                raise ValueError("Se requieren embeddings")

            # Preparar datos
            ids = [str(s['id']) for s in songs]
            metadatas = []

            for s in songs:
                metadata = {
                    'name': s.get('name', 'Unknown'),
                    'artist': s.get('artist', 'Unknown'),
                    'description': s.get('description', ''),
                    'genre': s.get('genre', 'unknown'),
                    'url': s.get('url', '')
                }
                metadatas.append(metadata)

            # Convertir embeddings a lista si es numpy array
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            elif isinstance(embeddings, list) and len(embeddings) > 0:
                if hasattr(embeddings[0], 'tolist'):
                    embeddings = [emb.tolist() for emb in embeddings]

            # Añadir a ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )

            print(f"✅ {len(songs)} canciones añadidas a la base de datos")

        except Exception as e:
            print(f"❌ Error al añadir canciones: {e}")
            raise

    def search(self, query_embedding, n_results=5, filter_dict=None):
        """
        Busca canciones similares al embedding de consulta

        Args:
            query_embedding (list/numpy.ndarray): Vector de embeddings de la consulta
            n_results (int): Número de resultados a retornar
            filter_dict (dict): Filtros opcionales (ej: {'genre': 'pop'})

        Returns:
            list: Lista de dicts con información de canciones encontradas
        """
        try:
            if self.collection.count() == 0:
                print("⚠️ La base de datos está vacía")
                return []

            # Convertir a lista si es numpy array
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()

            # Realizar búsqueda
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                where=filter_dict
            )

            # Formatear resultados
            formatted_results = []

            if results['metadatas'] and len(results['metadatas'][0]) > 0:
                for idx, metadata in enumerate(results['metadatas'][0]):
                    result = {
                        'id': results['ids'][0][idx],
                        'name': metadata.get('name', 'Unknown'),
                        'artist': metadata.get('artist', 'Unknown'),
                        'description': metadata.get('description', ''),
                        'genre': metadata.get('genre', 'unknown'),
                        'url': metadata.get('url', ''),
                        'distance': results['distances'][0][idx] if 'distances' in results else None,
                        'similarity': 1 - results['distances'][0][idx] if 'distances' in results else None
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return []

    def search_by_genre(self, query_embedding, genre, n_results=5):
        """
        Busca canciones filtrando por género

        Args:
            query_embedding: Vector de embeddings
            genre (str): Género a filtrar
            n_results (int): Número de resultados

        Returns:
            list: Canciones encontradas
        """
        return self.search(
            query_embedding,
            n_results=n_results,
            filter_dict={'genre': genre}
        )

    def get_song_by_id(self, song_id):
        """
        Obtiene información de una canción por su ID

        Args:
            song_id (str): ID de la canción

        Returns:
            dict: Información de la canción o None si no existe
        """
        try:
            result = self.collection.get(ids=[str(song_id)])

            if result['metadatas'] and len(result['metadatas']) > 0:
                metadata = result['metadatas'][0]
                return {
                    'id': song_id,
                    'name': metadata.get('name', 'Unknown'),
                    'artist': metadata.get('artist', 'Unknown'),
                    'description': metadata.get('description', ''),
                    'genre': metadata.get('genre', 'unknown'),
                    'url': metadata.get('url', '')
                }
            return None

        except Exception as e:
            print(f"❌ Error al obtener canción: {e}")
            return None

    def count(self):
        """
        Retorna el número de canciones en la base de datos
        """
        try:
            return self.collection.count()
        except Exception as e:
            print(f"❌ Error al contar canciones: {e}")
            return 0

    def get_all_genres(self):
        """
        Obtiene lista de todos los géneros en la base de datos
        """
        try:
            # Obtener todos los metadatos
            all_data = self.collection.get()
            genres = set()

            for metadata in all_data['metadatas']:
                if 'genre' in metadata:
                    genres.add(metadata['genre'])

            return sorted(list(genres))

        except Exception as e:
            print(f"❌ Error al obtener géneros: {e}")
            return []

    def delete_song(self, song_id):
        """
        Elimina una canción de la base de datos
        """
        try:
            self.collection.delete(ids=[str(song_id)])
            print(f"✅ Canción eliminada: {song_id}")
        except Exception as e:
            print(f"❌ Error al eliminar canción: {e}")

    def clear_all(self):
        """
        Elimina todas las canciones de la base de datos
        ⚠️ USAR CON PRECAUCIÓN
        """
        try:
            self.client.delete_collection("songs")
            self.collection = self.client.create_collection(
                name="songs",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Base de datos limpiada completamente")
        except Exception as e:
            print(f"❌ Error al limpiar base de datos: {e}")

    def get_stats(self):
        """
        Obtiene estadísticas de la base de datos
        """
        try:
            total_songs = self.count()
            genres = self.get_all_genres()

            return {
                'total_songs': total_songs,
                'total_genres': len(genres),
                'genres': genres,
                'collection_name': self.collection.name,
                'path': Config.CHROMA_DB_PATH
            }
        except Exception as e:
            print(f"❌ Error al obtener estadísticas: {e}")
            return {}

    def update_song_metadata(self, song_id, **kwargs):
        """
        Actualiza metadatos de una canción existente

        Args:
            song_id (str): ID de la canción
            **kwargs: Campos a actualizar (name, artist, description, genre, url)
        """
        try:
            # Obtener canción actual
            current = self.get_song_by_id(song_id)
            if not current:
                print(f"⚠️ Canción no encontrada: {song_id}")
                return False

            # Actualizar campos
            for key, value in kwargs.items():
                if key in ['name', 'artist', 'description', 'genre', 'url']:
                    current[key] = value

            # Actualizar en ChromaDB
            self.collection.update(
                ids=[str(song_id)],
                metadatas=[{k: v for k, v in current.items() if k != 'id'}]
            )

            print(f"✅ Canción actualizada: {song_id}")
            return True

        except Exception as e:
            print(f"❌ Error al actualizar canción: {e}")
            return False


# Testing
if __name__ == "__main__":
    print("🧪 Probando base de datos vectorial...")

    # Importar embeddings para testing
    from embeddings import EmbeddingModel

    embedder = EmbeddingModel()
    db = VectorDB()

    # Test 1: Añadir canciones de ejemplo
    print("\n📝 Test 1: Añadir canciones")

    test_songs = [
        {
            'id': 'test_1',
            'name': 'Happy Song',
            'artist': 'Joy Artist',
            'description': 'música alegre y energética para bailar',
            'genre': 'pop',
            'url': 'https://spotify.com/track/1'
        },
        {
            'id': 'test_2',
            'name': 'Sad Melody',
            'artist': 'Melancholic Band',
            'description': 'canción triste y melancólica para reflexionar',
            'genre': 'indie',
            'url': 'https://spotify.com/track/2'
        },
        {
            'id': 'test_3',
            'name': 'Workout Beats',
            'artist': 'Energy DJ',
            'description': 'música energética para hacer ejercicio y entrenar',
            'genre': 'electronic',
            'url': 'https://spotify.com/track/3'
        }
    ]

    # Generar embeddings
    descriptions = [s['description'] for s in test_songs]
    embeddings = embedder.encode_batch(descriptions)

    # Añadir a la base de datos
    db.add_songs(test_songs, embeddings)

    # Test 2: Buscar canciones
    print("\n🔍 Test 2: Buscar canciones similares")

    query = "necesito música para ponerme de buen humor"
    query_embedding = embedder.encode(query)

    results = db.search(query_embedding, n_results=3)

    print(f"Query: '{query}'")
    print(f"Resultados encontrados: {len(results)}")
    for result in results:
        print(f"  • {result['name']} - {result['artist']}")
        print(f"    Similitud: {result['similarity']:.3f}")
        print(f"    Género: {result['genre']}")

    # Test 3: Estadísticas
    print("\n📊 Test 3: Estadísticas")
    stats = db.get_stats()
    print(f"Total de canciones: {stats['total_songs']}")
    print(f"Géneros disponibles: {', '.join(stats['genres'])}")

    # Test 4: Buscar por género
    print("\n🎸 Test 4: Buscar por género")
    pop_results = db.search_by_genre(query_embedding, 'pop', n_results=2)
    print(f"Canciones de género 'pop': {len(pop_results)}")

    print("\n✅ Todos los tests completados")