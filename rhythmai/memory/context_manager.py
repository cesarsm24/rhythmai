"""
Gestor de contexto conversacional y perfil de usuario.

Módulo que coordina la memoria conversacional y el perfil de usuario
para proporcionar recomendaciones personalizadas basadas en historial.
"""

from .conversation_memory import ConversationMemory
from .user_profile import UserProfile


class ContextManager:
    """
    Gestor central del contexto conversacional.

    Combina memoria de conversación a corto plazo con perfil de usuario
    a largo plazo para personalizar las recomendaciones musicales.

    Attributes:
        user_id (str): Identificador único del usuario.
        conversation_memory (ConversationMemory): Memoria conversacional de corto plazo.
        user_profile (UserProfile): Perfil de usuario persistente de largo plazo.
    """

    def __init__(self, user_id="default_user"):
        """
        Inicializa el gestor de contexto para un usuario específico.

        Args:
            user_id (str): Identificador único del usuario.
        """
        self.user_id = user_id
        self.conversation_memory = ConversationMemory(user_id)
        self.user_profile = UserProfile(user_id)

    def add_interaction(self, user_text=None, emotion=None, genre=None, emotion_data=None):
        """
        Registra una nueva interacción del usuario en el sistema de memoria.

        Puede recibir un diccionario completo de análisis emocional o construir
        uno básico a partir de parámetros individuales.

        Args:
            user_text (str, optional): Texto de entrada del usuario.
            emotion (str, optional): Emoción dominante detectada.
            genre (str, optional): Género musical asociado.
            emotion_data (dict, optional): Diccionario completo de análisis emocional.

        Returns:
            dict: Interacción registrada con timestamp y metadata.
        """
        # Construir emotion_data si no se proporciona
        if emotion_data is None and emotion:
            emotion_data = {
                'dominant_emotion': emotion,
                'dominant_score': 1.0,
                'suggested_genres': [genre] if genre else ['pop'],
                'dimensions': {
                    'energy': 0.5,
                    'valence': 0.5
                }
            }

        # Registrar interacción en memoria conversacional
        interaction = self.conversation_memory.add_interaction(
            user_input=user_text,
            ai_response="",
            emotion_data=emotion_data,
            recommendations=None
        )

        # Actualizar estadísticas del perfil de usuario
        if emotion:
            self.user_profile.update_statistics(emotion=emotion)

        return interaction

    def get_enriched_context(self):
        """
        Obtiene contexto enriquecido combinando memoria conversacional y perfil de usuario.

        Returns:
            dict: Diccionario con las siguientes claves:
                - conversation_context: Resumen de conversaciones recientes
                - music_preferences: Preferencias musicales detectadas
                - emotion_history: Historial de emociones
                - user_preferences: Preferencias persistentes del usuario
        """
        default_context = {
            'conversation_context': "Esta es tu primera conversación.",
            'music_preferences': {
                'total_interactions': 0,
                'favorite_genres': [],
                'common_emotions': []
            },
            'emotion_history': [],
            'user_preferences': {}
        }

        try:
            conversation_context = self.conversation_memory.get_conversation_context()
            music_prefs = self.conversation_memory.get_music_preferences()

            if music_prefs is None:
                music_prefs = default_context['music_preferences']

            emotion_history = self.conversation_memory.get_emotion_history()
            user_prefs = self.user_profile.get_preferences()

            return {
                'conversation_context': conversation_context,
                'music_preferences': music_prefs,
                'emotion_history': emotion_history if emotion_history else [],
                'user_preferences': user_prefs if user_prefs else {}
            }
        except Exception as e:
            print(f"Error obteniendo contexto enriquecido: {e}")
            import traceback
            traceback.print_exc()
            return default_context

    def get_personalized_prompt(self, current_input):
        """
        Genera un prompt personalizado basado en el contexto del usuario.

        Incorpora conversaciones previas, preferencias musicales y
        tendencias emocionales para enriquecer la entrada actual.

        Args:
            current_input (str): Entrada actual del usuario.

        Returns:
            str: Prompt enriquecido con contexto personalizado.
        """
        context = self.get_enriched_context()

        prompt = f"Input actual del usuario: {current_input}\n\n"

        # Añadir contexto conversacional si existe
        if context['conversation_context'] != "Esta es tu primera conversación.":
            prompt += f"Contexto de conversaciones previas:\n{context['conversation_context']}\n\n"

        # Añadir preferencias musicales
        music_prefs = context.get('music_preferences')
        if music_prefs and music_prefs.get('favorite_genres'):
            top_genres = [g[0] for g in music_prefs['favorite_genres'][:3]]
            prompt += f"Géneros que suele disfrutar: {', '.join(top_genres)}\n"

            if music_prefs.get('common_emotions'):
                top_emotions = [e[0] for e in music_prefs['common_emotions'][:3]]
                prompt += f"Emociones frecuentes: {', '.join(top_emotions)}\n"

        # Añadir historial emocional reciente
        if context['emotion_history']:
            recent_emotions = [e['emotion'] for e in context['emotion_history'][-3:]]
            prompt += f"Emociones recientes: {', '.join(recent_emotions)}\n"

        return prompt

    def clear_all(self):
        """
        Limpia toda la memoria conversacional del usuario.

        Método útil para testing o para permitir al usuario reiniciar su historial.
        """
        self.conversation_memory.clear_history()