from .spotify_client import SpotifyClient
from .embeddings import EmbeddingModel
from .emotion_analyzer import EmotionAnalyzer
from .vector_db import VectorDB
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.memory.context_manager import ContextManager


class MusicRecommender:
    """
    Sistema de recomendación musical inteligente
    Combina análisis emocional + búsqueda semántica + API de Spotify + memoria conversacional
    """

    def __init__(self, user_id="default_user"):
        try:
            print("🎵 Inicializando sistema de recomendación musical...")

            # Componentes principales
            self.spotify = SpotifyClient()
            self.embedder = EmbeddingModel()
            self.emotion_analyzer = EmotionAnalyzer()
            self.vector_db = VectorDB()

            # 🧠 Sistema de memoria conversacional
            self.context_manager = ContextManager(user_id)

            print("✅ Sistema de recomendación inicializado correctamente")

        except Exception as e:
            print(f"❌ Error al inicializar sistema de recomendación: {e}")
            raise

    def recommend(self, user_input, n_results=8):
        """
        Sistema de recomendación híbrido con MEMORIA CONVERSACIONAL

        Args:
            user_input (str): Descripción del estado de ánimo del usuario
            n_results (int): Número de recomendaciones a retornar

        Returns:
            dict: Resultados completos con análisis, recomendaciones y contexto
        """
        try:
            print(f"🔍 Procesando: '{user_input[:50]}...'")

            # 1. Obtener contexto enriquecido de conversaciones previas
            enriched_context = self.context_manager.get_enriched_context()

            # 2. Análisis emocional detallado
            emotion_data = self.emotion_analyzer.analyze(user_input)
            print(f"😊 Emoción dominante: {emotion_data['dominant_emotion']} ({emotion_data['dominant_score']:.0%})")

            # 3. Ajustar recomendaciones según historial del usuario
            adjusted_params = self._adjust_with_history(emotion_data, enriched_context)

            # 4. Búsqueda semántica en base vectorial
            query_embedding = self.embedder.encode(user_input)
            vector_results = []

            if self.vector_db.count() > 0:
                vector_results = self.vector_db.search(query_embedding, n_results=3)
                print(f"💾 Encontradas {len(vector_results)} canciones en base vectorial")
            else:
                print("⚠️ Base de datos vectorial vacía")

            # 5. Recomendaciones de Spotify basadas en emociones
            spotify_recs = self._get_spotify_recommendations(adjusted_params, n_results)
            print(f"🎵 {len(spotify_recs)} recomendaciones de Spotify")

            # 6. Playlists contextuales
            context_suggestions = self._get_context_based_playlists(
                emotion_data['context'],
                emotion_data['dominant_emotion']
            )

            # 7. Generar explicación personalizada
            explanation = self._generate_personalized_explanation(
                emotion_data,
                enriched_context
            )

            # 8. GUARDAR EN MEMORIA para futuras recomendaciones
            ai_response = explanation
            track_names = [t['name'] for t in spotify_recs[:3]] if spotify_recs else []

            self.context_manager.add_interaction(
                user_input=user_input,
                ai_response=ai_response,
                emotion_data=emotion_data,
                recommendations=track_names
            )

            print("✅ Recomendación completada y guardada en memoria")

            return {
                'emotion_analysis': emotion_data,
                'vector_results': vector_results,
                'spotify_recommendations': spotify_recs,
                'context_playlists': context_suggestions,
                'explanation': explanation,
                'enriched_context': enriched_context
            }

        except Exception as e:
            print(f"❌ Error en recomendación: {e}")
            raise

    def _adjust_with_history(self, emotion_data, enriched_context):
        """
        Ajusta parámetros de recomendación según historial del usuario
        Suaviza cambios bruscos y considera preferencias previas
        """
        params = emotion_data['spotify_params'].copy()
        genres = emotion_data['suggested_genres'].copy()

        # Si tiene preferencias previas, incorporarlas
        if enriched_context.get('music_preferences'):
            prefs = enriched_context['music_preferences']

            # Priorizar géneros favoritos si existen
            if prefs.get('favorite_genres'):
                top_genre = prefs['favorite_genres'][0][0]
                if top_genre not in genres:
                    # Insertar género favorito al inicio
                    genres.insert(0, top_genre)
                    print(f"🎸 Añadido género favorito: {top_genre}")

        # Suavizar cambios bruscos basándose en historial emocional
        if enriched_context.get('emotion_history'):
            recent_emotions = enriched_context['emotion_history'][-3:]  # Últimas 3

            if recent_emotions:
                # Calcular promedios de energía y valencia recientes
                recent_energy = [e['energy'] for e in recent_emotions if 'energy' in e]
                recent_valence = [e['valence'] for e in recent_emotions if 'valence' in e]

                if recent_energy:
                    avg_energy = sum(recent_energy) / len(recent_energy)
                    # Suavizar: 70% actual, 30% promedio histórico
                    params['target_energy'] = params['target_energy'] * 0.7 + avg_energy * 0.3
                    print(f"⚡ Energía ajustada: {params['target_energy']:.2f}")

                if recent_valence:
                    avg_valence = sum(recent_valence) / len(recent_valence)
                    params['target_valence'] = params['target_valence'] * 0.7 + avg_valence * 0.3
                    print(f"😊 Valencia ajustada: {params['target_valence']:.2f}")

        # Actualizar géneros en params
        params['genres'] = genres

        return params

    def _get_spotify_recommendations(self, params, n_results):
        """
        Obtiene recomendaciones de Spotify API basadas en parámetros emocionales
        """
        spotify_recs = []

        try:
            genres = params.get('genres', ['pop'])[:2]  # Max 2 géneros como seed

            # Llamada a Spotify API con parámetros detallados
            recs = self.spotify.sp.recommendations(
                seed_genres=genres if genres else ['pop'],
                target_energy=params['target_energy'],
                target_valence=params['target_valence'],
                target_danceability=params.get('target_danceability', 0.5),
                target_acousticness=params.get('target_acousticness', 0.5),
                min_tempo=params.get('tempo_range', [80, 120])[0],
                max_tempo=params.get('tempo_range', [80, 120])[1],
                limit=n_results
            )

            for track in recs['tracks']:
                spotify_recs.append({
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'url': track['external_urls']['spotify'],
                    'uri': track['uri'],
                    'preview_url': track.get('preview_url'),
                    'album_image': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'source': 'spotify_emotion_match'
                })

        except Exception as e:
            print(f"⚠️ Error en Spotify API: {e}")

            # Fallback: búsqueda simple por texto
            try:
                dominant_emotion = params.get('dominant_emotion', 'happy')
                fallback_query = f"{dominant_emotion} music"
                tracks = self.spotify.search_track(fallback_query, limit=n_results)
                spotify_recs = tracks
                print(f"🔄 Usando búsqueda fallback con query: '{fallback_query}'")
            except Exception as e2:
                print(f"❌ Fallback también falló: {e2}")

        return spotify_recs

    def _get_context_based_playlists(self, contexts, dominant_emotion):
        """
        Sugiere playlists públicas de Spotify según contexto detectado
        """
        suggestions = []

        context_queries = {
            'study/work': f"{dominant_emotion} study focus music",
            'workout': f"{dominant_emotion} workout gym music",
            'relax/sleep': f"{dominant_emotion} relaxing calm music",
            'party': f"{dominant_emotion} party dance music",
            'driving': f"{dominant_emotion} driving road trip music",
            'emotional_release': f"{dominant_emotion} emotional cathartic music",
            'morning': f"{dominant_emotion} morning wake up music",
            'night': f"{dominant_emotion} night evening music",
        }

        for context in contexts[:2]:  # Max 2 contextos
            if context in context_queries:
                try:
                    query = context_queries[context]
                    results = self.spotify.sp.search(q=query, type='playlist', limit=2)

                    for item in results['playlists']['items']:
                        suggestions.append({
                            'name': item['name'],
                            'url': item['external_urls']['spotify'],
                            'context': context,
                            'description': item.get('description', '')
                        })
                except Exception as e:
                    print(f"⚠️ Error buscando playlists para contexto '{context}': {e}")

        return suggestions

    def _generate_personalized_explanation(self, emotion_data, enriched_context):
        """
        Genera explicación personalizada considerando memoria conversacional
        """
        explanation = "🎵 "

        # Mencionar si es usuario recurrente
        if enriched_context.get('music_preferences'):
            prefs = enriched_context['music_preferences']
            total = prefs.get('total_interactions', 0)

            if total > 10:
                explanation += f"Basándome en nuestras {total} conversaciones anteriores y "
            elif total > 5:
                explanation += "Considerando nuestro historial juntos y "
            elif total > 1:
                explanation += "Recordando nuestras charlas previas y "
            else:
                explanation += "Basándome en "
        else:
            explanation += "Basándome en "

        # Análisis emocional actual
        dominant = emotion_data['dominant_emotion']
        explanation += f"que te sientes principalmente **{dominant}**, "

        energy = emotion_data['dimensions']['energy']
        valence = emotion_data['dimensions']['valence']

        if energy > 0.7:
            explanation += "con mucha energía, "
        elif energy < 0.3:
            explanation += "con poca energía, "

        # Mencionar patrones si existen
        if enriched_context.get('emotion_history'):
            recent_emotions = [e['emotion'] for e in enriched_context['emotion_history'][-3:]]
            if recent_emotions.count(dominant) >= 2:
                explanation += f"(noto que has estado {dominant} últimamente) "

        if valence > 0.6:
            explanation += "te he seleccionado música positiva y edificante"
        elif valence < 0.4:
            explanation += "te he seleccionado música que te acompañe en este momento"
        else:
            explanation += "te he seleccionado música equilibrada"

        # Mencionar géneros favoritos si existen
        if enriched_context.get('music_preferences'):
            fav_genres = enriched_context['music_preferences'].get('favorite_genres', [])
            if fav_genres:
                top_genres = [g[0] for g in fav_genres[:2]]
                explanation += f". Incluí algunos de tus géneros favoritos: {', '.join(top_genres)}"

        # Géneros sugeridos actuales
        current_genres = emotion_data['suggested_genres'][:3]
        explanation += f". Géneros sugeridos: {', '.join(current_genres)}."

        return explanation

    def get_conversation_summary(self):
        """
        Obtiene resumen completo de la conversación y preferencias
        """
        return self.context_manager.get_enriched_context()

    def get_user_stats(self):
        """
        Obtiene estadísticas del usuario
        """
        context = self.context_manager.get_enriched_context()

        stats = {
            'total_interactions': 0,
            'favorite_genres': [],
            'common_emotions': [],
            'recent_mood_trend': 'neutral'
        }

        if context.get('music_preferences'):
            prefs = context['music_preferences']
            stats['total_interactions'] = prefs.get('total_interactions', 0)
            stats['favorite_genres'] = prefs.get('favorite_genres', [])[:5]
            stats['common_emotions'] = prefs.get('common_emotions', [])[:5]

        # Analizar tendencia de humor reciente
        if context.get('emotion_history'):
            recent = context['emotion_history'][-5:]
            avg_valence = sum(e['valence'] for e in recent) / len(recent) if recent else 0.5

            if avg_valence > 0.6:
                stats['recent_mood_trend'] = 'positivo'
            elif avg_valence < 0.4:
                stats['recent_mood_trend'] = 'negativo'
            else:
                stats['recent_mood_trend'] = 'neutral'

        return stats


# Testing
if __name__ == "__main__":
    print("🧪 Probando sistema de recomendación...")

    try:
        recommender = MusicRecommender(user_id="test_user")

        # Test con diferentes inputs
        test_inputs = [
            "Me siento muy feliz y con energía, quiero bailar",
            "Estoy estudiando y necesito concentrarme",
            "Hoy ha sido un día difícil, estoy triste"
        ]

        for test_input in test_inputs:
            print(f"\n📝 Input: '{test_input}'")
            results = recommender.recommend(test_input, n_results=5)

            print(f"✅ Emoción: {results['emotion_analysis']['dominant_emotion']}")
            print(f"   Explicación: {results['explanation']}")
            print(f"   Recomendaciones: {len(results['spotify_recommendations'])}")

        # Mostrar estadísticas finales
        print("\n📊 Estadísticas del usuario:")
        stats = recommender.get_user_stats()
        print(f"   Total interacciones: {stats['total_interactions']}")
        print(f"   Tendencia de humor: {stats['recent_mood_trend']}")

        print("\n✅ Tests completados correctamente")

    except Exception as e:
        print(f"❌ Error en testing: {e}")
        import traceback

        traceback.print_exc()