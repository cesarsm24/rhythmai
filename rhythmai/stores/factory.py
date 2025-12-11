"""
Factory para crear instancias de vector stores

Proporciona una función centralizada para crear el vector store
configurado en las variables de entorno.
"""

import logging
from rhythmai.config import Config

logger = logging.getLogger(__name__)


def get_vector_store():
    """
    Crea y retorna una instancia del vector store configurado

    Lee la configuración de Config.VECTOR_STORE y retorna la
    implementación correspondiente (ChromaDB o FAISS).

    Returns:
        BaseVectorStore: Instancia del vector store configurado

    Raises:
        ValueError: Si VECTOR_STORE contiene un valor inválido
    """
    store_type = Config.VECTOR_STORE.lower()

    if store_type == "chroma":
        from rhythmai.stores.chroma_store import ChromaStore
        logger.info("Usando ChromaDB como vector store")
        return ChromaStore()

    elif store_type == "faiss":
        from rhythmai.stores.faiss_store import FAISSStore
        logger.info("Usando FAISS como vector store")
        return FAISSStore()

    else:
        raise ValueError(
            f"Vector store inválido: '{store_type}'. "
            f"Opciones válidas: 'chroma', 'faiss'"
        )