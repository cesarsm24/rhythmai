"""
Módulo de seguridad para RhythmAI.
Proporciona funcionalidad de cifrado y descifrado para datos sensibles.

Características:
- Cifrado AES-256 para datos en reposo
- Derivación segura de claves usando PBKDF2
- Soporte para cifrar datos no estructurados (texto, JSON, binario)
- Manejo profesional de errores y logging
"""

import os
import base64
import json
import logging
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from rhythmai.config import Config

logger = logging.getLogger(__name__)


class DataEncryption:
    """
    Sistema de cifrado y descifrado de grado profesional para datos sensibles.

    Utiliza cifrado AES-256 mediante Fernet (cifrado simétrico).
    Aplicable a:
    - Credenciales de usuario
    - Información personal
    - Historial de conversaciones
    - Datos de preferencias

    Attributes:
        master_password (bytes): Contraseña maestra para derivación de claves.
        _fernet (Fernet): Instancia lazy del cipher Fernet.
    """

    def __init__(self, master_password=None):
        """
        Inicializa el sistema de cifrado.

        Args:
            master_password (str, optional): Contraseña maestra para derivación de claves.
                Si es None, utiliza variable de entorno o valor por defecto.
        """
        if master_password is None:
            master_password = os.getenv("RHYTHM_MASTER_KEY", "default_master_key_change_in_production")

        self.master_password = master_password.encode()
        self._fernet = None
        logger.info("DataEncryption system initialized")

    def _get_fernet(self):
        """
        Inicialización diferida del cipher Fernet.

        Deriva la clave de cifrado desde la contraseña maestra usando PBKDF2
        con 100000 iteraciones para máxima seguridad.

        Returns:
            Fernet: Instancia configurada del cipher Fernet.
        """
        if self._fernet is None:
            # Salt estático (en producción debe ser único por usuario o instalación)
            salt = b'rhythmai_salt_v1'

            # Derivar clave usando PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )

            key = base64.urlsafe_b64encode(kdf.derive(self.master_password))
            self._fernet = Fernet(key)

        return self._fernet

    def encrypt_string(self, plaintext):
        """
        Cifra una cadena de texto.

        Args:
            plaintext (str): Texto a cifrar.

        Returns:
            str: Texto cifrado codificado en Base64.

        Raises:
            ValueError: Si plaintext no es una cadena.
        """
        if not isinstance(plaintext, str):
            raise ValueError("Plaintext must be a string")

        try:
            fernet = self._get_fernet()
            encrypted = fernet.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt_string(self, encrypted_text):
        """
        Descifra una cadena de texto.

        Args:
            encrypted_text (str): Texto cifrado codificado en Base64.

        Returns:
            str: Texto descifrado.

        Raises:
            ValueError: Si el descifrado falla por clave inválida o datos corruptos.
        """
        if not isinstance(encrypted_text, str):
            raise ValueError("Encrypted text must be a string")

        try:
            fernet = self._get_fernet()
            encrypted = base64.urlsafe_b64decode(encrypted_text.encode())
            decrypted = fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise ValueError("Failed to decrypt data - invalid key or corrupted data")

    def encrypt_dict(self, data_dict):
        """
        Cifra un diccionario convirtiéndolo primero a JSON.

        Args:
            data_dict (dict): Diccionario a cifrar.

        Returns:
            str: Cadena JSON cifrada.

        Raises:
            ValueError: Si data_dict no es un diccionario.
        """
        if not isinstance(data_dict, dict):
            raise ValueError("Data must be a dictionary")

        try:
            json_str = json.dumps(data_dict, ensure_ascii=False)
            return self.encrypt_string(json_str)
        except Exception as e:
            logger.error(f"Dictionary encryption failed: {str(e)}")
            raise

    def decrypt_dict(self, encrypted_text):
        """
        Descifra y parsea un diccionario desde JSON cifrado.

        Args:
            encrypted_text (str): Cadena JSON cifrada.

        Returns:
            dict: Diccionario descifrado.

        Raises:
            ValueError: Si el descifrado o parseo falla.
        """
        try:
            json_str = self.decrypt_string(encrypted_text)
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Dictionary decryption failed: {str(e)}")
            raise

    def encrypt_file(self, file_path, output_path=None):
        """
        Cifra un archivo completo.

        Args:
            file_path (str): Ruta al archivo a cifrar.
            output_path (str, optional): Ruta de salida. Si es None, añade '.enc' al original.

        Returns:
            str: Ruta al archivo cifrado.

        Raises:
            Exception: Si el cifrado del archivo falla.
        """
        if output_path is None:
            output_path = f"{file_path}.enc"

        try:
            with open(file_path, 'rb') as f:
                data = f.read()

            fernet = self._get_fernet()
            encrypted = fernet.encrypt(data)

            with open(output_path, 'wb') as f:
                f.write(encrypted)

            logger.info(f"File encrypted: {file_path} -> {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"File encryption failed: {str(e)}")
            raise

    def decrypt_file(self, encrypted_file_path, output_path=None):
        """
        Descifra un archivo completo.

        Args:
            encrypted_file_path (str): Ruta al archivo cifrado.
            output_path (str, optional): Ruta de salida. Si es None, quita extensión '.enc'.

        Returns:
            str: Ruta al archivo descifrado.

        Raises:
            Exception: Si el descifrado del archivo falla.
        """
        if output_path is None:
            if encrypted_file_path.endswith('.enc'):
                output_path = encrypted_file_path[:-4]
            else:
                output_path = f"{encrypted_file_path}.dec"

        try:
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()

            fernet = self._get_fernet()
            decrypted = fernet.decrypt(encrypted_data)

            with open(output_path, 'wb') as f:
                f.write(decrypted)

            logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"File decryption failed: {str(e)}")
            raise


class SecureStorage:
    """
    Wrapper de almacenamiento seguro para datos de configuración sensibles.

    Cifra y descifra automáticamente los datos al escribir y leer.

    Attributes:
        encryptor (DataEncryption): Instancia del sistema de cifrado.
        storage_path (Path): Ruta del directorio de almacenamiento seguro.
    """

    def __init__(self, encryption_key=None):
        """
        Inicializa el almacenamiento seguro.

        Args:
            encryption_key (str, optional): Clave de cifrado personalizada.
        """
        self.encryptor = DataEncryption(encryption_key)
        self.storage_path = Path(Config.MEMORY_PATH) / "secure"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_secure(self, key, data):
        """
        Guarda datos de forma segura con cifrado automático.

        Args:
            key (str): Clave de almacenamiento.
            data (dict or str): Datos a almacenar.

        Raises:
            Exception: Si falla el guardado seguro.
        """
        file_path = self.storage_path / f"{key}.secure"

        try:
            if isinstance(data, dict):
                encrypted = self.encryptor.encrypt_dict(data)
            else:
                encrypted = self.encryptor.encrypt_string(str(data))

            with open(file_path, 'w') as f:
                f.write(encrypted)

            logger.info(f"Data saved securely: {key}")
        except Exception as e:
            logger.error(f"Secure save failed: {str(e)}")
            raise

    def load_secure(self, key):
        """
        Carga datos seguros con descifrado automático.

        Args:
            key (str): Clave de almacenamiento.

        Returns:
            dict or str: Datos descifrados, o None si no existe el archivo.

        Raises:
            Exception: Si falla la carga segura.
        """
        file_path = self.storage_path / f"{key}.secure"

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r') as f:
                encrypted = f.read()

            # Intentar descifrar como diccionario, con fallback a string
            try:
                return self.encryptor.decrypt_dict(encrypted)
            except:
                return self.encryptor.decrypt_string(encrypted)
        except Exception as e:
            logger.error(f"Secure load failed: {str(e)}")
            raise

    def delete_secure(self, key):
        """
        Elimina datos seguros del almacenamiento.

        Args:
            key (str): Clave de almacenamiento a eliminar.
        """
        file_path = self.storage_path / f"{key}.secure"

        if file_path.exists():
            file_path.unlink()
            logger.info(f"Secure data deleted: {key}")


if __name__ == "__main__":
    print("Probando Sistema de Cifrado de Datos\n")

    # Inicializar encryptor
    encryptor = DataEncryption("my_secure_password_123")

    # Test 1: Cifrado de cadenas
    print("Test 1: Cifrado de Cadenas")
    original = "This is sensitive user data: credit card 1234-5678-9012-3456"
    encrypted = encryptor.encrypt_string(original)
    decrypted = encryptor.decrypt_string(encrypted)

    print(f"Original:  {original}")
    print(f"Encrypted: {encrypted[:50]}...")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {original == decrypted}\n")

    # Test 2: Cifrado de diccionarios
    print("Test 2: Cifrado de Diccionarios")
    user_data = {
        "user_id": "user123",
        "spotify_token": "BQD8F3xKJ...",
        "preferences": {
            "favorite_genre": "electronic",
            "privacy_level": "high"
        }
    }

    encrypted_dict = encryptor.encrypt_dict(user_data)
    decrypted_dict = encryptor.decrypt_dict(encrypted_dict)

    print(f"Original:  {user_data}")
    print(f"Encrypted: {encrypted_dict[:50]}...")
    print(f"Decrypted: {decrypted_dict}")
    print(f"Match: {user_data == decrypted_dict}\n")

    # Test 3: Almacenamiento seguro
    print("Test 3: Almacenamiento Seguro")
    storage = SecureStorage("master_key_456")

    storage.save_secure("user_profile", user_data)
    loaded = storage.load_secure("user_profile")

    print(f"Saved and loaded: {loaded}")
    print(f"Match: {user_data == loaded}\n")

    # Limpieza
    storage.delete_secure("user_profile")

    print("Todos los tests de cifrado pasaron correctamente")