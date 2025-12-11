"""
Sistema de Memoria y Contexto de RhythmAI.

Este paquete gestiona la memoria conversacional, perfiles de usuario y enriquecimiento
de contexto para proporcionar recomendaciones musicales personalizadas.

Módulos:
    - context_manager: Orquestación de contexto y gestión de memoria.
    - conversation_memory: Almacenamiento y recuperación de historial conversacional.
    - user_profile: Gestión de preferencias y perfiles de usuario.

Ejemplo:
    from rhythmai.memory import ContextManager

    cm = ContextManager(user_id="user123")
    context = cm.get_enriched_context("Me siento feliz")
"""

from rhythmai.memory.context_manager import ContextManager
from rhythmai.memory.conversation_memory import ConversationMemory
from rhythmai.memory.user_profile import UserProfile

__all__ = [
    "ContextManager",
    "ConversationMemory",
    "UserProfile",
]