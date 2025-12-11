"""
Sistema inteligente de recomendación musical.

Implementa búsqueda vectorial semántica con filtrado por género
para generación de recomendaciones musicales personalizadas.
"""

import logging
import random

from rhythmai.core.deezer_client import DeezerClient
from rhythmai.core.embeddings import EmbeddingModel
from rhythmai.core.emotion_analyzer import EmotionAnalyzer
from rhythmai.stores.factory import get_vector_store

logger = logging.getLogger(__name__)


class MusicRecommender:
    """
    Sistema inteligente de recomendación musical basado en análisis emocional.

    Combina análisis de emociones, búsqueda vectorial semántica y filtrado
    por género para generar recomendaciones musicales personalizadas.
    """

    def __init__(self, user_id="default_user"):
        """
        Inicializa el sistema de recomendación.

        Args:
            user_id (str): Identificador único del usuario.

        Raises:
            Exception: Si falla la inicialización de algún componente crítico.
        """
        try:
            logger.info(f"Inicializando sistema de recomendación para usuario: {user_id}")

            self.deezer = DeezerClient()
            self.embedder = EmbeddingModel()
            # Pasar el modelo embedder para reutilizar y evitar carga duplicada en memoria
            self.emotion_analyzer = EmotionAnalyzer(embedder=self.embedder.model)
            self.vector_store = get_vector_store()

            from rhythmai.memory.context_manager import ContextManager
            self.context_manager = ContextManager(user_id=user_id)

            logger.info("Sistema de recomendación inicializado exitosamente")
            logger.info(f"Base de datos vectorial contiene {self.vector_store.count()} canciones")

        except Exception as e:
            logger.error(f"Error al inicializar sistema de recomendación: {e}")
            raise

    def recommend(self, user_input, n_results=8, randomize=False):
        """
        Genera recomendaciones musicales personalizadas basadas en entrada del usuario.

        Proceso de recomendación:
        1. Análisis emocional del texto de entrada
        2. Enriquecimiento semántico de la consulta
        3. Búsqueda vectorial con filtrado por género
        4. Generación de explicación simple
        5. Opcionalmente randomiza resultados para mayor variedad

        Args:
            user_input (str): Descripción del estado emocional o preferencias del usuario.
            n_results (int): Número de resultados deseados. Por defecto 8.
            randomize (bool): Si True, introduce variación aleatoria en los resultados.

        Returns:
            dict: Diccionario con las siguientes claves:
                - emotion_analysis: Análisis emocional completo
                - music_recommendations: Recomendaciones musicales
                - vector_results: Resultados de búsqueda vectorial local
                - context_playlists: Listas de reproducción contextuales
                - explanation: Explicación simple de las recomendaciones
                - enriched_context: Contexto enriquecido del usuario
        """
        logger.info(f"Procesando solicitud de recomendación: '{user_input[:50]}...'")

        enriched_context = self._get_safe_context()
        emotion_data = self.emotion_analyzer.analyze(user_input)

        if not emotion_data or "dimensions" not in emotion_data:
            logger.error("Estructura de emotion_data inválida, usando respuesta neutral")
            emotion_data = self.emotion_analyzer.get_default_emotion_response(confidence=0.50)

        logger.info(
            f"Emoción dominante detectada: {emotion_data['dominant_emotion']} "
            f"(confianza: {emotion_data['dominant_score']:.0%})"
        )

        if emotion_data.get('explicit_genres'):
            logger.info(f"Géneros explícitos detectados: {emotion_data['explicit_genres']}")
        logger.info(f"Géneros sugeridos finales: {emotion_data['suggested_genres'][:3]}")

        vector_results = []
        if self.vector_store and self.vector_store.count() > 0:
            try:
                enriched_query = self._create_enriched_query(user_input, emotion_data)
                query_embedding = self.embedder.encode(enriched_query)

                suggested_genres = emotion_data.get('suggested_genres', [])
                primary_genre = suggested_genres[0] if suggested_genres else None

                # Aumentar resultados si randomización está activa para mayor variedad
                search_n = n_results * 2 if randomize else n_results

                if primary_genre:
                    logger.info(f"Filtrando por género principal: '{primary_genre}'")
                    logger.info(f"Query enriquecida: '{enriched_query}'")

                    vector_results = self.vector_store.search(
                        query_embedding,
                        n_results=search_n,
                        filter_dict={'genre': primary_genre}
                    )

                    logger.info(
                        f"Encontrados {len(vector_results)} resultados en género '{primary_genre}'"
                    )

                    # Búsqueda complementaria en género secundario si resultados insuficientes
                    if len(vector_results) < search_n // 2 and len(suggested_genres) > 1:
                        secondary_genre = suggested_genres[1]
                        logger.warning(
                            f"Solo {len(vector_results)} resultados con '{primary_genre}', "
                            f"buscando también en '{secondary_genre}'"
                        )

                        secondary_results = self.vector_store.search(
                            query_embedding,
                            n_results=search_n - len(vector_results),
                            filter_dict={'genre': secondary_genre}
                        )

                        vector_results.extend(secondary_results)
                        logger.info(
                            f"Añadidos {len(secondary_results)} resultados de '{secondary_genre}'"
                        )

                    if len(vector_results) < 3:
                        logger.warning(
                            f"Solo {len(vector_results)} resultados totales. "
                            f"La base de datos puede tener insuficientes canciones "
                            f"del género '{primary_genre}'"
                        )
                else:
                    logger.info("Sin género específico detectado, búsqueda sin filtro")
                    vector_results = self.vector_store.search(
                        query_embedding,
                        n_results=search_n
                    )
                    logger.info(f"Encontrados {len(vector_results)} resultados sin filtro")

                # Aplicar randomización si está activada
                if randomize and len(vector_results) > n_results:
                    # Mantener resultados más relevantes y randomizar el resto
                    top_results = vector_results[:n_results // 2]
                    remaining = vector_results[n_results // 2:]
                    random.shuffle(remaining)
                    vector_results = top_results + remaining[:n_results - len(top_results)]
                    logger.info(f"Resultados randomizados: {len(vector_results)} canciones")
                elif len(vector_results) > n_results:
                    vector_results = vector_results[:n_results]

            except Exception as e:
                logger.error(f"Error en búsqueda vectorial: {e}")
                import traceback
                logger.error(traceback.format_exc())

        explanation = self._generate_simple_explanation(emotion_data)

        # Registrar interacción en el contexto del usuario
        try:
            self.context_manager.add_interaction(
                user_text=user_input,
                emotion_data=emotion_data
            )
        except Exception as e:
            logger.error(f"Error al guardar interacción: {e}")

        return {
            "emotion_analysis": emotion_data,
            "music_recommendations": [],
            "vector_results": vector_results,
            "context_playlists": [],
            "explanation": explanation,
            "enriched_context": enriched_context
        }

    def _create_enriched_query(self, original_text, emotion_data):
        """
        Crea una consulta enriquecida con información emocional.

        Añade descriptores emocionales y musicales al texto original
        para mejorar la calidad del matching semántico en la búsqueda vectorial.

        Args:
            original_text (str): Texto original del usuario.
            emotion_data (dict): Análisis emocional del texto.

        Returns:
            str: Consulta enriquecida con descriptores adicionales.
        """
        enriched = original_text
        emotion = emotion_data.get('dominant_emotion', 'neutral')

        # Descriptores emocionales para enriquecer la búsqueda
        emotion_descriptors = {
            'sadness': 'música triste melancólica emotiva',
            'grief': 'música triste emotiva para procesar dolor',
            'joy': 'música alegre feliz positiva',
            'excitement': 'música emocionante energética',
            'anger': 'música intensa agresiva',
            'love': 'música romántica amorosa',
            'fear': 'música tranquila calmante',
            'chill': 'música relajante tranquila',
            'neutral': 'música'
        }

        emotion_desc = emotion_descriptors.get(emotion, f'música {emotion}')
        enriched += f" {emotion_desc}"

        # Añadir descriptores basados en dimensiones emocionales
        dimensions = emotion_data.get('dimensions', {})
        energy = dimensions.get('energy', 0.5)
        valence = dimensions.get('valence', 0.5)

        if energy > 0.7:
            enriched += " energética intensa potente"
        elif energy < 0.3:
            enriched += " tranquila suave calmada"

        if valence > 0.7:
            enriched += " alegre positiva"
        elif valence < 0.3:
            enriched += " melancólica emotiva"

        return enriched

    def _get_safe_context(self):
        """
        Obtiene el contexto del usuario con manejo robusto de errores.

        Returns:
            dict: Contexto enriquecido del usuario o contexto por defecto
                  en caso de error.
        """
        try:
            enriched = self.context_manager.get_enriched_context()

            if not isinstance(enriched, dict):
                raise ValueError("Estructura de contexto inválida")

            if "music_preferences" not in enriched:
                enriched["music_preferences"] = {"total_interactions": 0}

            return enriched

        except Exception as e:
            logger.warning(f"Error al recuperar contexto: {e}")
            return {
                "music_preferences": {
                    "total_interactions": 0,
                    "favorite_genres": [],
                    "favorite_emotions": []
                }
            }

    def _adjust_with_history(self, emotion_data, enriched_context):
        """
        Ajusta parámetros de recomendación basándose en historial del usuario.

        Incorpora géneros favoritos del usuario en la lista de géneros sugeridos
        si no están ya presentes.

        Args:
            emotion_data (dict): Análisis emocional actual.
            enriched_context (dict): Contexto enriquecido del usuario.

        Returns:
            dict: Parámetros ajustados para la recomendación.
        """
        params = emotion_data.get("music_params", {}).copy()
        genres = emotion_data.get("suggested_genres", ["pop"]).copy()

        prefs = enriched_context.get("music_preferences", {})

        # Incorporar género favorito del usuario si existe
        if prefs and prefs.get("favorite_genres"):
            favorite_genre = prefs["favorite_genres"][0][0]
            if favorite_genre not in genres:
                genres.insert(0, favorite_genre)
                logger.info(f"Añadido género favorito del usuario: {favorite_genre}")

        params["genres"] = genres
        params["dominant_emotion"] = emotion_data["dominant_emotion"]

        return params

    def _generate_simple_explanation(self, emotion_data):
        """
        Genera explicación simple de las recomendaciones.

        Args:
            emotion_data (dict): Análisis emocional del usuario.

        Returns:
            str: Explicación breve y directa de las recomendaciones.
        """
        emotion = emotion_data.get("dominant_emotion", "neutral")
        dimensions = emotion_data.get("dimensions", {"valence": 0.5, "energy": 0.5})
        energy = dimensions.get("energy", 0.5)

        text = f"Música para cuando te sientes {emotion}"

        if energy < 0.3:
            text += ", con ritmo suave"
        elif energy > 0.7:
            text += ", con mucha energía"

        return text + "."