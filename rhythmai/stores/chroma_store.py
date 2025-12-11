"""
Implementación de vector store usando ChromaDB.

ChromaDB es una base de datos vectorial open-source optimizada
para almacenamiento y búsqueda de embeddings.
"""

import chromadb
from chromadb.config import Settings
import logging

from rhythmai.config import Config
from rhythmai.stores.base_store import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):
    """
    Vector store basado en ChromaDB.

    Características principales:
    - Búsqueda por similitud coseno
    - Soporte para filtrado por metadata
    - Persistencia en disco
    - Índice HNSW para búsquedas rápidas

    Attributes:
        client (chromadb.PersistentClient): Cliente persistente de ChromaDB.
        collection (chromadb.Collection): Colección de canciones.
    """

    def __init__(self):
        """
        Inicializa el vector store de ChromaDB.

        Raises:
            Exception: Si falla la inicialización de ChromaDB.
        """
        try:
            # Crear cliente persistente
            self.client = chromadb.PersistentClient(
                path=Config.CHROMA_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Obtener o crear colección con distancia coseno
            self.collection = self.client.get_or_create_collection(
                name="songs",
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"ChromaDB inicializado: {Config.CHROMA_DB_PATH}")
            logger.info(f"Canciones en base de datos: {self.collection.count()}")

        except Exception as e:
            logger.error(f"Error al inicializar ChromaDB: {e}")
            raise

    def add_songs(self, songs, embeddings):
        """
        Añade canciones a la base de datos.

        Args:
            songs (list): Lista de diccionarios con información de canciones:
                - id: Identificador único
                - name: Nombre de la canción
                - artist: Artista
                - description: Descripción o mood
                - genre: Género musical
                - url: URL de Deezer
                - album_image: URL de la portada del álbum
                - preview_url: URL del preview de audio
            embeddings (np.ndarray): Array de embeddings (n_songs × 384).

        Raises:
            ValueError: Si los datos de entrada son inválidos.
        """
        try:
            # Validar datos de entrada
            if not songs or len(songs) == 0:
                raise ValueError("Se requieren canciones")

            if embeddings is None or (hasattr(embeddings, 'shape') and embeddings.shape[0] == 0):
                raise ValueError("Se requieren embeddings")

            # Preparar identificadores
            ids = [str(song['id']) for song in songs]
            metadatas = []

            # Preparar metadata (ChromaDB no acepta valores None)
            for song in songs:
                metadata = {
                    'name': str(song.get('name') or 'Unknown'),
                    'artist': str(song.get('artist') or 'Unknown'),
                    'description': str(song.get('description') or ''),
                    'genre': str(song.get('genre') or 'unknown'),
                    'url': str(song.get('url') or ''),
                    'album_image': str(song.get('album_image') or ''),
                    'preview_url': str(song.get('preview_url') or '')
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

            logger.info(f"{len(songs)} canciones añadidas a ChromaDB")

        except Exception as e:
            logger.error(f"Error al añadir canciones: {e}")
            raise

    def search(self, query_embedding, n_results=5, filter_dict=None):
        """
        Busca canciones similares por embedding.

        Args:
            query_embedding: Vector de embeddings de consulta.
            n_results (int): Número de resultados a retornar. Por defecto 5.
            filter_dict (dict, optional): Filtros de metadata (ejemplo: {'genre': 'pop'}).

        Returns:
            list: Lista de diccionarios con información de canciones encontradas.
        """
        try:
            if self.collection.count() == 0:
                logger.warning("Base de datos vacía")
                return []

            # Convertir a lista si es numpy array
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()

            # Realizar búsqueda vectorial
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                where=filter_dict
            )

            # Formatear resultados
            formatted_results = []

            if results['metadatas'] and len(results['metadatas'][0]) > 0:
                for idx, metadata in enumerate(results['metadatas'][0]):
                    # Convertir strings vacíos a None para mejor manejo
                    album_image = metadata.get('album_image', '')
                    preview_url = metadata.get('preview_url', '')

                    result = {
                        'id': results['ids'][0][idx],
                        'name': metadata.get('name', 'Unknown'),
                        'artist': metadata.get('artist', 'Unknown'),
                        'description': metadata.get('description', ''),
                        'genre': metadata.get('genre', 'unknown'),
                        'url': metadata.get('url', ''),
                        'album_image': album_image if album_image else None,
                        'preview_url': preview_url if preview_url else None,
                        'distance': results['distances'][0][idx] if 'distances' in results else None,
                        'similarity': 1 - results['distances'][0][idx] if 'distances' in results else None
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []

    def count(self):
        """
        Retorna el número de canciones en la base de datos.

        Returns:
            int: Número total de canciones almacenadas.
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error al contar canciones: {e}")
            return 0

    def get_all_genres(self):
        """
        Obtiene lista de todos los géneros en la base de datos.

        Returns:
            list: Lista ordenada alfabéticamente de géneros únicos.
        """
        try:
            all_data = self.collection.get()
            genres = set()

            for metadata in all_data['metadatas']:
                if 'genre' in metadata:
                    genres.add(metadata['genre'])

            return sorted(list(genres))

        except Exception as e:
            logger.error(f"Error al obtener géneros: {e}")
            return []

    def clear_all(self):
        """
        Elimina todas las canciones de la base de datos.

        Operación irreversible que elimina completamente la colección
        y la recrea vacía.
        """
        try:
            self.client.delete_collection("songs")
            self.collection = self.client.create_collection(
                name="songs",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Base de datos ChromaDB limpiada completamente")

        except Exception as e:
            logger.error(f"Error al limpiar base de datos: {e}")

    def get_stats(self):
        """
        Obtiene estadísticas de la base de datos.

        Returns:
            dict: Diccionario con estadísticas incluyendo:
                - total_songs: Número total de canciones
                - total_genres: Número de géneros únicos
                - genres: Lista de géneros
                - collection_name: Nombre de la colección
                - path: Ruta de almacenamiento
                - store_type: Tipo de vector store
        """
        try:
            total_songs = self.count()
            genres = self.get_all_genres()

            return {
                'total_songs': total_songs,
                'total_genres': len(genres),
                'genres': genres,
                'collection_name': self.collection.name,
                'path': Config.CHROMA_DB_PATH,
                'store_type': 'ChromaDB'
            }

        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {e}")
            return {}