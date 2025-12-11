"""
Script para limpiar el historial de memoria del sistema.

Incluye limpieza de archivos cifrados (.secure, .enc) y directorio secure/.
"""

import os
import shutil

from rhythmai.config import Config
from rhythmai.memory.context_manager import ContextManager


def clear_all_users():
    """
    Elimina todos los archivos de memoria del sistema.

    Busca y elimina archivos de historial, base de datos y archivos cifrados
    en el directorio de memoria configurado.

    Incluye:
    - Archivos .json (historial y perfiles, cifrados o no)
    - Archivos .db (bases de datos)
    - Archivos .secure (almacenamiento cifrado de SecureStorage)
    - Archivos .enc (archivos cifrados)
    - Directorio secure/ completo
    """
    memory_path = Config.MEMORY_PATH

    if not os.path.exists(memory_path):
        print(f"No hay archivos de memoria en {memory_path}")
        return

    files_deleted = 0

    # Eliminar archivos de memoria (incluye archivos cifrados)
    for file in os.listdir(memory_path):
        filepath = os.path.join(memory_path, file)

        # Saltar directorios (se manejan después)
        if os.path.isdir(filepath):
            continue

        # Eliminar archivos de memoria
        if file.endswith(('.json', '.db', '.secure', '.enc')):
            try:
                os.remove(filepath)
                print(f"Eliminado: {file}")
                files_deleted += 1
            except Exception as e:
                print(f"Error eliminando {file}: {e}")

    # Eliminar directorio secure/ si existe
    secure_path = os.path.join(memory_path, 'secure')
    if os.path.exists(secure_path):
        try:
            shutil.rmtree(secure_path)
            print(f"Eliminado directorio: secure/")
            files_deleted += 1
        except Exception as e:
            print(f"Error eliminando directorio secure/: {e}")

    print(f"\nTotal archivos/directorios eliminados: {files_deleted}")


def clear_specific_user(user_id):
    """
    Elimina la memoria de un usuario específico.

    Args:
        user_id (str): Identificador del usuario.
    """
    cm = ContextManager(user_id=user_id)
    cm.clear_all()
    print(f"Memoria limpiada para usuario: {user_id}")


if __name__ == "__main__":
    print("Limpiador de Memoria - RhythmAI\n")
    print("Opciones:")
    print("1. Limpiar TODO el historial (todos los usuarios)")
    print("2. Limpiar usuario específico")
    print("3. Cancelar")

    choice = input("\nSeleccione una opción (1-3): ").strip()

    if choice == "1":
        confirm = input("Advertencia: Esta operación eliminará TODO el historial (s/n): ").strip().lower()
        if confirm == 's':
            clear_all_users()
        else:
            print("Operación cancelada")

    elif choice == "2":
        user_id = input("Introduzca el user_id (default: streamlit_user): ").strip()
        if not user_id:
            user_id = "streamlit_user"
        clear_specific_user(user_id)

    else:
        print("Operación cancelada")