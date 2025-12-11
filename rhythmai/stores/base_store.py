"""
Clase base abstracta para vector stores

Define la interfaz común que deben implementar todos los vector stores.
"""

from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """
    Interfaz abstracta para bases de datos vectoriales

    Define los métodos que deben implementar todas las implementaciones
    de vector stores (ChromaDB, FAISS, etc.)
    """

    @abstractmethod
    def add_songs(self, songs, embeddings):
        """
        Añade canciones a la base de datos

        Args:
            songs (list): Lista de diccionarios con información de canciones
            embeddings (np.ndarray): Array de embeddings (n_songs × dimension)
        """
        pass

    @abstractmethod
    def search(self, query_embedding, n_results=5, filter_dict=None):
        """
        Busca canciones similares

        Args:
            query_embedding: Vector de embeddings de consulta
            n_results (int): Número de resultados a retornar
            filter_dict (dict, optional): Filtros de metadata

        Returns:
            list: Lista de canciones encontradas con scores de similitud
        """
        pass

    @abstractmethod
    def count(self):
        """
        Retorna el número de canciones en la base de datos

        Returns:
            int: Número total de canciones almacenadas
        """
        pass

    @abstractmethod
    def get_all_genres(self):
        """
        Obtiene lista de todos los géneros en la base de datos

        Returns:
            list: Lista de géneros únicos
        """
        pass

    @abstractmethod
    def clear_all(self):
        """
        Elimina todas las canciones de la base de datos

        Advertencia: Esta operación es irreversible
        """
        pass

    @abstractmethod
    def get_stats(self):
        """
        Obtiene estadísticas de la base de datos

        Returns:
            dict: Estadísticas (total canciones, géneros, etc.)
        """
        pass