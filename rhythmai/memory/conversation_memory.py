"""
Módulo de memoria conversacional.

Gestiona el historial de conversaciones sin dependencias externas.
Almacena interacciones del usuario, datos emocionales y recomendaciones.

El historial de conversaciones se almacena cifrado con AES-256 para seguridad.
"""

import json
import os
from datetime import datetime

from rhythmai.config import Config
from rhythmai.utils.security import DataEncryption


class ConversationMemory:
    """
    Sistema de memoria conversacional simplificado.

    Mantiene historial de conversaciones y contexto sin dependencias
    de frameworks externos como LangChain.

    Attributes:
        user_id (str): Identificador único del usuario.
        memory_path (str): Ruta al archivo de historial del usuario.
        encryptor (DataEncryption): Instancia del sistema de cifrado.
    """

    def __init__(self, user_id="default_user"):
        """
        Inicializa la memoria conversacional para un usuario.

        Args:
            user_id (str): Identificador único del usuario.
        """
        self.user_id = user_id
        self.memory_path = os.path.join(Config.MEMORY_PATH, f"{user_id}_history.json")

        # Inicializar sistema de cifrado
        self.encryptor = DataEncryption()

        os.makedirs(Config.MEMORY_PATH, exist_ok=True)

        print(f"Memoria inicializada para usuario: {user_id}")

    def add_interaction(self, user_input, ai_response, emotion_data=None, recommendations=None):
        """
        Añade una nueva interacción a la memoria conversacional.

        Args:
            user_input (str): Texto de entrada del usuario.
            ai_response (str): Respuesta generada por el sistema.
            emotion_data (dict, optional): Análisis emocional de la entrada.
            recommendations (list, optional): Lista de recomendaciones generadas.

        Returns:
            dict: Interacción registrada con timestamp y metadata completa.
        """
        timestamp = datetime.now().isoformat()

        interaction = {
            'timestamp': timestamp,
            'user_input': user_input,
            'ai_response': ai_response,
            'emotion_data': emotion_data,
            'recommendations': recommendations
        }

        self._save_interaction(interaction)

        return interaction

    def get_recent_interactions(self, n=5):
        """
        Obtiene las últimas N interacciones del historial.

        Args:
            n (int): Número de interacciones a recuperar.

        Returns:
            list: Lista de las últimas N interacciones ordenadas cronológicamente.
        """
        history = self._load_full_history()
        return history[-n:] if history else []

    def get_conversation_context(self, max_tokens=500):
        """
        Obtiene el contexto conversacional formateado como texto.

        Args:
            max_tokens (int): Límite aproximado de tokens para el contexto.

        Returns:
            str: Resumen textual de las conversaciones recientes.
        """
        recent = self.get_recent_interactions(n=Config.MEMORY_WINDOW)

        if not recent:
            return "Esta es tu primera conversación."

        context = "Historial de conversación reciente:\n\n"

        for i, interaction in enumerate(recent, 1):
            user_text = interaction['user_input'][:100]
            context += f"[{i}] Usuario: {user_text}...\n"

            if interaction.get('emotion_data'):
                emotion = interaction['emotion_data'].get('dominant_emotion', 'unknown')
                context += f"    Emoción detectada: {emotion}\n"
            context += "\n"

        # Limitar longitud del contexto
        if len(context) > max_tokens * 4:
            context = context[:max_tokens * 4] + "..."

        return context

    def get_emotion_history(self, n=10):
        """
        Obtiene el historial de emociones detectadas en las interacciones.

        Args:
            n (int): Número de interacciones a analizar.

        Returns:
            list: Lista de diccionarios con datos emocionales por interacción.
        """
        history = self.get_recent_interactions(n)
        emotions = []

        for interaction in history:
            if interaction.get('emotion_data'):
                emotions.append({
                    'timestamp': interaction['timestamp'],
                    'emotion': interaction['emotion_data'].get('dominant_emotion'),
                    'score': interaction['emotion_data'].get('dominant_score'),
                    'energy': interaction['emotion_data']['dimensions'].get('energy'),
                    'valence': interaction['emotion_data']['dimensions'].get('valence')
                })

        return emotions

    def get_music_preferences(self):
        """
        Analiza preferencias musicales basándose en el historial completo.

        Calcula frecuencias de géneros y emociones para identificar patrones
        en las preferencias del usuario.

        Returns:
            dict: Diccionario con géneros favoritos, emociones comunes y estadísticas,
                  o None si no hay historial disponible.
        """
        history = self._load_full_history()

        if not history:
            return None

        all_genres = []
        all_emotions = []

        # Extraer géneros y emociones de todas las interacciones
        for interaction in history:
            if interaction.get('emotion_data'):
                emotion = interaction['emotion_data'].get('dominant_emotion')
                if emotion:
                    all_emotions.append(emotion)

                genres = interaction['emotion_data'].get('suggested_genres', [])
                all_genres.extend(genres)

        from collections import Counter

        genre_counts = Counter(all_genres)
        emotion_counts = Counter(all_emotions)

        return {
            'favorite_genres': genre_counts.most_common(5),
            'common_emotions': emotion_counts.most_common(5),
            'total_interactions': len(history)
        }

    def clear_history(self):
        """
        Elimina completamente el historial de conversaciones del usuario.
        """
        if os.path.exists(self.memory_path):
            os.remove(self.memory_path)
            print("Historial limpiado correctamente")

    def _save_interaction(self, interaction):
        """
        Guarda una interacción en el archivo de historial cifrado.

        Mantiene solo las últimas MAX_CONVERSATION_HISTORY interacciones
        para evitar crecimiento ilimitado del archivo.

        Args:
            interaction (dict): Interacción a guardar.
        """
        history = self._load_full_history()
        history.append(interaction)

        # Mantener solo las últimas interacciones según configuración
        if len(history) > Config.MAX_CONVERSATION_HISTORY:
            history = history[-Config.MAX_CONVERSATION_HISTORY:]

        # Cifrar y guardar el historial completo
        encrypted_data = self.encryptor.encrypt_dict({'history': history})
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            f.write(encrypted_data)

    def _load_full_history(self):
        """
        Carga el historial completo desde el archivo cifrado.

        Intenta descifrar el contenido. Si falla, asume formato antiguo sin cifrar
        para mantener compatibilidad hacia atrás y lo migra al formato cifrado.

        Returns:
            list: Lista de todas las interacciones guardadas, o lista vacía si no existe.
        """
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Intentar descifrar (formato actual)
                try:
                    data = self.encryptor.decrypt_dict(content)
                    return data.get('history', [])
                except (ValueError, KeyError) as e:
                    # Formato antiguo sin cifrar (compatibilidad hacia atrás)
                    try:
                        history = json.loads(content)
                        # Migrar a formato cifrado
                        encrypted_data = self.encryptor.encrypt_dict({'history': history})
                        with open(self.memory_path, 'w', encoding='utf-8') as f:
                            f.write(encrypted_data)
                        print(f"Historial migrado a formato cifrado para usuario {self.user_id}")
                        return history
                    except (json.JSONDecodeError, ValueError) as parse_error:
                        print(f"Error parseando historial: {parse_error}")
                        return []
            except Exception as e:
                print(f"Error cargando historial: {e}")
                return []
        return []