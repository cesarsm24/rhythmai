"""
Módulos de bases de datos vectoriales

Proporciona implementaciones de vector stores para búsqueda semántica:
- ChromaStore: Basado en ChromaDB
- FAISSStore: Basado en FAISS (Facebook AI Similarity Search)

Incluye factory pattern para selección dinámica del store.
"""

from rhythmai.stores.base_store import BaseVectorStore
from rhythmai.stores.chroma_store import ChromaStore
from rhythmai.stores.faiss_store import FAISSStore
from rhythmai.stores.factory import get_vector_store

# Alias for backward compatibility
create_vector_store = get_vector_store

__all__ = [
    "BaseVectorStore",
    "ChromaStore",
    "FAISSStore",
    "get_vector_store",
    "create_vector_store",
]