"""
Módulo de perfil de usuario.

Gestiona preferencias persistentes del usuario y su historial de escucha.
Almacena géneros favoritos, patrones emocionales y estadísticas de uso.

Los perfiles se almacenan cifrados con AES-256 para seguridad.
"""

import json
import os
from datetime import datetime

from rhythmai.config import Config
from rhythmai.utils.security import DataEncryption


class UserProfile:
    """
    Perfil del usuario con preferencias persistentes.

    Mantiene información a largo plazo sobre las preferencias musicales
    y emocionales del usuario, así como estadísticas de uso del sistema.

    Attributes:
        user_id (str): Identificador único del usuario.
        profile_path (str): Ruta al archivo de perfil del usuario.
        encryptor (DataEncryption): Instancia del sistema de cifrado.
        profile (dict): Datos del perfil cargados en memoria.
    """

    def __init__(self, user_id="default_user"):
        """
        Inicializa el perfil de usuario, cargándolo desde disco si existe.

        Args:
            user_id (str): Identificador único del usuario.
        """
        self.user_id = user_id
        self.profile_path = os.path.join(Config.MEMORY_PATH, f"{user_id}_profile.json")

        # Inicializar sistema de cifrado
        self.encryptor = DataEncryption()

        os.makedirs(Config.MEMORY_PATH, exist_ok=True)

        self.profile = self._load_profile()

    def _load_profile(self):
        """
        Carga el perfil desde archivo o crea uno nuevo si no existe.

        Intenta descifrar el perfil. Si falla, asume formato antiguo sin cifrar
        para mantener compatibilidad hacia atrás y lo migra al formato cifrado.

        Returns:
            dict: Datos del perfil del usuario.
        """
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Intentar descifrar (formato actual)
                try:
                    return self.encryptor.decrypt_dict(content)
                except (ValueError, KeyError) as e:
                    # Formato antiguo sin cifrar (compatibilidad hacia atrás)
                    try:
                        profile = json.loads(content)
                        # Migrar a formato cifrado
                        self._save_profile_internal(profile)
                        print(f"Perfil migrado a formato cifrado para usuario {self.user_id}")
                        return profile
                    except (json.JSONDecodeError, ValueError) as parse_error:
                        print(f"Error parseando perfil: {parse_error}")
                        return self._create_default_profile()
            except (IOError, OSError) as e:
                print(f"Error leyendo archivo de perfil: {e}")
                return self._create_default_profile()
        return self._create_default_profile()

    def _create_default_profile(self):
        """
        Crea un perfil por defecto con estructura inicial.

        Returns:
            dict: Perfil de usuario con valores por defecto.
        """
        return {
            'user_id': self.user_id,
            'created_at': datetime.now().isoformat(),
            'preferences': {
                'favorite_genres': [],
                'disliked_genres': [],
                'preferred_energy_range': [0.3, 0.7],
                'preferred_valence_range': [0.3, 0.7],
                'language': 'es'
            },
            'statistics': {
                'total_sessions': 0,
                'total_recommendations': 0,
                'most_common_emotion': None,
                'last_session': None
            },
            'listening_history': []
        }

    def update_preferences(self, **kwargs):
        """
        Actualiza las preferencias del usuario.

        Args:
            **kwargs: Pares clave-valor de preferencias a actualizar.
        """
        for key, value in kwargs.items():
            if key in self.profile['preferences']:
                self.profile['preferences'][key] = value

        self._save_profile()

    def add_to_history(self, track_info):
        """
        Añade una canción al historial de escucha del usuario.

        Mantiene solo las últimas 100 canciones para evitar crecimiento excesivo.

        Args:
            track_info (dict): Información de la canción escuchada.
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'track': track_info
        }

        self.profile['listening_history'].append(entry)

        # Limitar historial a las últimas 100 entradas
        if len(self.profile['listening_history']) > 100:
            self.profile['listening_history'] = self.profile['listening_history'][-100:]

        self._save_profile()

    def update_statistics(self, emotion=None):
        """
        Actualiza las estadísticas de uso del sistema.

        Args:
            emotion (str, optional): Emoción dominante de la sesión actual.
        """
        self.profile['statistics']['total_sessions'] += 1
        self.profile['statistics']['last_session'] = datetime.now().isoformat()

        if emotion:
            self.profile['statistics']['most_common_emotion'] = emotion

        self._save_profile()

    def get_preferences(self):
        """
        Obtiene las preferencias actuales del usuario.

        Returns:
            dict: Diccionario con todas las preferencias configuradas.
        """
        return self.profile['preferences']

    def _save_profile(self):
        """
        Guarda el perfil en disco cifrado con AES-256.
        """
        self._save_profile_internal(self.profile)

    def _save_profile_internal(self, profile_data):
        """
        Método interno para guardar perfil cifrado en disco.

        Args:
            profile_data (dict): Datos del perfil a guardar.
        """
        encrypted_data = self.encryptor.encrypt_dict(profile_data)
        with open(self.profile_path, 'w', encoding='utf-8') as f:
            f.write(encrypted_data)