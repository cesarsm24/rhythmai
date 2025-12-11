"""
Script para limpiar completamente la base de datos vectorial.

Operación irreversible que elimina todas las canciones almacenadas.
Incluye limpieza física completa de ChromaDB.
"""

import sys
import os
import shutil
from pathlib import Path
from rhythmai.stores.factory import get_vector_store
from rhythmai.config import Config

print("Limpiando base de datos vectorial...\n")

# Obtener store actual
db = get_vector_store()
print(f"Vector store: {type(db).__name__}")
print(f"Canciones actuales: {db.count()}")

# Modo no interactivo con flag --yes
auto_confirm = "--yes" in sys.argv or "-y" in sys.argv

if not auto_confirm:
    response = input("\n¿Desea eliminar TODA la base de datos? (s/n): ")
    if response.lower() not in ['sí', 'si', 'yes', 's', 'y']:
        print("Operación cancelada")
        sys.exit(0)

print("\nLimpiando...")

# Determinar qué store estamos usando
if Config.VECTOR_STORE == "chroma":
    print("Limpieza física de ChromaDB...")

    try:
        # Cerrar conexión si existe
        if hasattr(db, 'client'):
            del db.client

        # Eliminar directorio completo
        chroma_path = Path(Config.CHROMA_DB_PATH)
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            print(f"  ✓ Directorio {Config.CHROMA_DB_PATH} eliminado completamente")

        # Recrear estructura limpia
        chroma_path.mkdir(parents=True, exist_ok=True)
        print("  ✓ Directorio ChromaDB recreado vacío")

        # Reinicializar vector store
        from rhythmai.stores.chroma_store import ChromaStore
        new_db = ChromaStore()
        print(f"  ✓ ChromaDB reinicializado. Canciones: {new_db.count()}")

    except Exception as e:
        print(f"  ✗ Error en limpieza física: {e}")
        sys.exit(1)

elif Config.VECTOR_STORE == "faiss":
    print("Limpieza de FAISS...")

    try:
        # Eliminar archivos de FAISS
        faiss_path = Path(Config.FAISS_DB_PATH)
        if faiss_path.exists():
            shutil.rmtree(faiss_path)
            print(f"  ✓ Directorio {Config.FAISS_DB_PATH} eliminado")

        # Recrear
        faiss_path.mkdir(parents=True, exist_ok=True)
        print("  ✓ Directorio FAISS recreado vacío")

        # Reinicializar
        from rhythmai.stores.faiss_store import FAISSStore
        new_db = FAISSStore()
        print(f"  ✓ FAISS reinicializado. Canciones: {new_db.count()}")

    except Exception as e:
        print(f"  ✗ Error en limpieza FAISS: {e}")
        sys.exit(1)

else:
    # Fallback para otros stores
    db.clear_all()
    print(f"  ✓ Base de datos limpiada con método estándar")

print(f"\n✅ Base de datos completamente limpiada")