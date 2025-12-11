"""
Implementación de vector store usando FAISS.

FAISS (Facebook AI Similarity Search) es una biblioteca optimizada
para búsqueda de similitud ultra-rápida en vectores densos.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
import logging

from rhythmai.config import Config
from rhythmai.stores.base_store import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSStore(BaseVectorStore):
    """
    Vector store basado en FAISS.

    Ventajas sobre ChromaDB:
    - 10-100x más rápido en búsquedas
    - Menor uso de memoria
    - Escalable a millones de vectores

    Desventajas:
    - No tiene filtrado de metadata nativo
    - Requiere gestión manual de índices
    - Metadata se almacena separadamente

    Attributes:
        dimension (int): Dimensión de los vectores de embeddings.
        index_type (str): Tipo de índice FAISS utilizado.
        db_path (Path): Ruta del directorio de almacenamiento.
        index (faiss.Index): Índice FAISS para búsqueda vectorial.
        metadata (list): Lista de metadata de canciones.
        id_to_idx (dict): Mapeo de IDs a índices en FAISS.
    """

    def __init__(self, dimension=None, index_type="Flat"):
        """
        Inicializa el vector store de FAISS.

        Args:
            dimension (int): Dimensión de los vectores. Por defecto 384.
            index_type (str): Tipo de índice ("Flat", "IVF", "HNSW").

        Raises:
            Exception: Si falla la inicialización.
            ValueError: Si el tipo de índice es inválido.
        """
        try:
            # Usar dimensión de configuración si no se especifica
            self.dimension = dimension if dimension is not None else Config.EMBEDDING_DIMENSION
            self.index_type = index_type
            self.db_path = Path(Config.FAISS_DB_PATH)
            self.db_path.mkdir(parents=True, exist_ok=True)

            # Rutas para persistencia
            self.index_file = self.db_path / "faiss.index"
            self.metadata_file = self.db_path / "metadata.pkl"

            # Crear o cargar índice
            if self.index_file.exists():
                self._load_index()
            else:
                self._create_index()

            logger.info(f"FAISS VectorStore inicializado ({self.count()} vectores)")

        except Exception as e:
            logger.error(f"Error al inicializar FAISS: {e}")
            raise

    def _create_index(self):
        """
        Crea un nuevo índice FAISS según el tipo especificado.

        Raises:
            ValueError: Si el tipo de índice no es válido.
        """
        if self.index_type == "Flat":
            # Distancia L2 (Euclidiana) - Equivalente a coseno para vectores normalizados
            self.index = faiss.IndexFlatL2(self.dimension)

        elif self.index_type == "IVF":
            # Índice con Inverted File (más rápido para millones de vectores)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)

        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World (el más rápido)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)

        else:
            raise ValueError(f"Tipo de índice inválido: {self.index_type}")

        # Inicializar almacenamiento de metadata
        self.metadata = []
        self.id_to_idx = {}

        logger.info(f"Índice FAISS creado: {self.index_type}, dimension={self.dimension}")

    def _load_index(self):
        """
        Carga un índice existente desde disco.

        Si falla la carga, crea un nuevo índice.
        """
        try:
            self.index = faiss.read_index(str(self.index_file))

            with open(self.metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.id_to_idx = data['id_to_idx']

            logger.info(f"Índice FAISS cargado desde {self.index_file}")

        except Exception as e:
            logger.warning(f"Error cargando índice: {e}. Creando nuevo...")
            self._create_index()

    def _save_index(self):
        """
        Guarda el índice y metadata en disco.
        """
        try:
            faiss.write_index(self.index, str(self.index_file))

            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_to_idx': self.id_to_idx
                }, f)

            logger.info(f"Índice FAISS guardado en {self.index_file}")

        except Exception as e:
            logger.error(f"Error guardando índice: {e}")

    def add_songs(self, songs, embeddings):
        """
        Añade canciones al índice.

        Args:
            songs (list): Lista de diccionarios con información de canciones:
                - id: Identificador único
                - name: Nombre de la canción
                - artist: Artista
                - description: Descripción o mood
                - genre: Género musical
                - url: URL de Deezer
                - album_name: Nombre del álbum
                - album_image: URL de la portada del álbum
                - preview_url: URL del preview de audio
            embeddings (np.ndarray): Array de embeddings (n_songs × dimension).

        Raises:
            ValueError: Si los datos de entrada son inválidos.
        """
        try:
            if not songs or len(songs) == 0:
                raise ValueError("Se requieren canciones")

            # Convertir embeddings a numpy si es necesario
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Normalizar vectores para usar distancia L2 como similitud coseno
            faiss.normalize_L2(embeddings)

            # Añadir al índice
            start_idx = len(self.metadata)
            self.index.add(embeddings.astype('float32'))

            # Guardar metadata
            for idx, song in enumerate(songs):
                song_id = song.get('id', f"song_{start_idx + idx}")

                # Normalizar campos opcionales
                album_image = song.get('album_image', '')
                preview_url = song.get('preview_url', '')

                self.id_to_idx[song_id] = start_idx + idx
                self.metadata.append({
                    'id': song_id,
                    'name': song.get('name', 'Unknown'),
                    'artist': song.get('artist', 'Unknown'),
                    'description': song.get('description', ''),
                    'genre': song.get('genre', 'unknown'),
                    'url': song.get('url', ''),
                    'album_image': album_image if album_image else '',
                    'preview_url': preview_url if preview_url else ''
                })

            # Guardar cambios
            self._save_index()

            logger.info(f"{len(songs)} canciones añadidas a FAISS")

        except Exception as e:
            logger.error(f"Error añadiendo canciones: {e}")
            raise

    def search(self, query_embedding, n_results=5, filter_dict=None):
        """
        Busca vectores similares en el índice.

        Args:
            query_embedding (np.ndarray): Vector de búsqueda.
            n_results (int): Número de resultados a retornar. Por defecto 5.
            filter_dict (dict, optional): Filtros de metadata (ejemplo: {'genre': 'pop'}).

        Returns:
            list: Lista de resultados con metadata y scores de similitud.
        """
        try:
            if self.count() == 0:
                logger.warning("Base de datos vacía")
                return []

            # Convertir a numpy y normalizar
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)

            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)

            # Buscar más resultados si hay filtros para compensar el filtrado
            search_k = min(n_results * 2 if filter_dict else n_results, self.count())
            distances, indices = self.index.search(query_embedding, search_k)

            # Formatear resultados
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue

                metadata = self.metadata[idx]

                # Aplicar filtros si existen
                if filter_dict:
                    if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                        continue

                # Convertir distancia L2 a similitud (para vectores normalizados)
                similarity = 1.0 / (1.0 + dist)

                # Convertir strings vacíos a None para mejor manejo
                album_image = metadata.get('album_image', '')
                preview_url = metadata.get('preview_url', '')

                result = {
                    'id': metadata['id'],
                    'name': metadata['name'],
                    'artist': metadata['artist'],
                    'description': metadata['description'],
                    'genre': metadata['genre'],
                    'url': metadata['url'],
                    'album_image': album_image if album_image else None,
                    'preview_url': preview_url if preview_url else None,
                    'distance': float(dist),
                    'similarity': float(similarity)
                }
                results.append(result)

                if len(results) >= n_results:
                    break

            return results

        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []

    def count(self):
        """
        Retorna el número de vectores en el índice.

        Returns:
            int: Número total de canciones almacenadas.
        """
        return self.index.ntotal if hasattr(self, 'index') else 0

    def get_all_genres(self):
        """
        Obtiene lista de todos los géneros en la base de datos.

        Returns:
            list: Lista ordenada alfabéticamente de géneros únicos.
        """
        genres = set()
        for meta in self.metadata:
            if 'genre' in meta:
                genres.add(meta['genre'])
        return sorted(list(genres))

    def get_stats(self):
        """
        Obtiene estadísticas de la base de datos.

        Returns:
            dict: Diccionario con estadísticas incluyendo:
                - total_songs: Número total de canciones
                - total_genres: Número de géneros únicos
                - genres: Lista de géneros
                - index_type: Tipo de índice FAISS
                - dimension: Dimensión de los vectores
                - path: Ruta de almacenamiento
                - store_type: Tipo de vector store
        """
        return {
            'total_songs': self.count(),
            'total_genres': len(self.get_all_genres()),
            'genres': self.get_all_genres(),
            'index_type': self.index_type,
            'dimension': self.dimension,
            'path': str(self.db_path),
            'store_type': 'FAISS'
        }

    def clear_all(self):
        """
        Elimina todo el contenido del índice.

        Operación irreversible que recrea el índice completamente vacío.
        """
        self._create_index()
        self._save_index()
        logger.info("Índice FAISS limpiado completamente")