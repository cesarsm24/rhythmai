"""
Utilidades de RhythmAI.

Este paquete contiene módulos de utilidad para el sistema:
    - security: Cifrado/descifrado AES-256 para datos sensibles

Modulos de seguridad:
    - DataEncryption: Sistema de cifrado/descifrado con AES-256 y PBKDF2
    - SecureStorage: Almacenamiento automático cifrado de datos

Ejemplo:
    from rhythmai.utils import DataEncryption

    encryptor = DataEncryption()
    encrypted = encryptor.encrypt_string("datos sensibles")
    decrypted = encryptor.decrypt_string(encrypted)
"""

from rhythmai.utils.security import DataEncryption, SecureStorage

__all__ = [
    "DataEncryption",
    "SecureStorage",
]