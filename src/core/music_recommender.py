from .spotify_client import SpotifyClient
from .embeddings import EmbeddingModel
from .emotion_analyzer import EmotionAnalyzer
from .vector_db import VectorDB
import sys
from pathlib import Path

# Añadir directorio raíz
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.memory.context_manager import ContextManager


class MusicRecommender:
    """
    Sistema de recomendación musical inteligente.
    Combina:
    - Análisis emocional
    - Ajuste por memoria
    - Spotify API real
    - Fallback inteligente
    """

    def __init__(self, user_id="default_user"):
        try:
            print("🎵 Inicializando sistema de recomendación musical...")

            self.spotify = SpotifyClient()
            self.embedder = EmbeddingModel()
            self.emotion_analyzer = EmotionAnalyzer()
            self.vector_db = VectorDB()

            self.context_manager = ContextManager(user_id=user_id)

            print("✅ Sistema de recomendación inicializado correctamente")

        except Exception as e:
            print(f"❌ Error al inicializar sistema: {e}")
            raise

    # ====================================================================================
    # RECOMENDACIÓN PRINCIPAL
    # ====================================================================================
    def recommend(self, user_input, n_results=8):

        print(f"🔍 Procesando: '{user_input[:50]}...'")

        # 1) Memoria contextual
        enriched = self.context_manager.get_enriched_context()

        # 2) Análisis emocional
        emotion_data = self.emotion_analyzer.analyze(user_input)

        print(f"😊 Emoción dominante: {emotion_data['dominant_emotion']} ({emotion_data['dominant_score']:.0%})")

        # 3) Ajuste con historial
        params = self._adjust_with_history(emotion_data, enriched)

        # 4) Vector Search
        embedding = self.embedder.encode(user_input)
        vector_results = []

        if self.vector_db.count() > 0:
            vector_results = self.vector_db.search(embedding, n_results=3)
        else:
            print("⚠️ Base vectorial vacía")

        # 5) Recomendaciones Spotify
        spotify_recs = self._get_spotify_recommendations(params, n_results)
        print(f"🎵 {len(spotify_recs)} recomendaciones de Spotify")

        # 6) Playlists según contexto
        playlists = self._get_context_based_playlists(
            emotion_data["context"],
            emotion_data["dominant_emotion"]
        )

        # 7) Explicación personalizada
        explanation = self._generate_personalized_explanation(
            emotion_data,
            enriched
        )

        # 8) Guardar memoria
        genre = emotion_data["suggested_genres"][0] if emotion_data["suggested_genres"] else None

        self.context_manager.add_interaction(
            user_text=user_input,
            emotion=emotion_data["dominant_emotion"],
            genre=genre
        )

        return {
            "emotion_analysis": emotion_data,
            "spotify_recommendations": spotify_recs,
            "vector_results": vector_results,
            "context_playlists": playlists,
            "explanation": explanation,
            "enriched_context": enriched
        }

    # ====================================================================================
    # AJUSTE POR HISTORIAL
    # ====================================================================================
    def _adjust_with_history(self, emotion_data, enriched):

        params = emotion_data["spotify_params"].copy()
        genres = emotion_data["suggested_genres"].copy()

        prefs = enriched.get("music_preferences")

        if prefs and prefs.get("favorite_genres"):
            fav = prefs["favorite_genres"][0][0]
            if fav not in genres:
                genres.insert(0, fav)
                print(f"🎸 Añadiendo género favorito del usuario: {fav}")

        params["genres"] = genres
        params["dominant_emotion"] = emotion_data["dominant_emotion"]

        return params

    # ====================================================================================
    # SPOTIFY REAL + FALLBACK ARREGLADO
    # ====================================================================================
    def _get_spotify_recommendations(self, params, n_results):

        allowed_seeds = [
            "pop", "rock", "metal", "dance", "house", "edm", "electronic",
            "acoustic", "folk", "indie-pop", "hip-hop", "r-n-b", "soul",
            "latin", "reggaeton", "techno"
        ]

        seeds = []
        for g in params.get("genres", []):
            g = g.lower().replace(" ", "-")
            if g in allowed_seeds:
                seeds.append(g)

        if not seeds:
            seeds = ["pop"]

        seeds = seeds[:2]

        try:
            recs = self.spotify.sp.recommendations(
                seed_genres=seeds,
                target_energy=params.get("target_energy", 0.5),
                target_valence=params.get("target_valence", 0.5),
                min_tempo=params["tempo_range"][0],
                max_tempo=params["tempo_range"][1],
                limit=n_results
            )

            results = []
            for t in recs["tracks"]:
                results.append({
                    "name": t["name"],
                    "artist": t["artists"][0]["name"],
                    "url": t["external_urls"]["spotify"],
                    "uri": t["uri"],
                    "preview_url": t.get("preview_url"),
                    "album_image": t["album"]["images"][0]["url"]
                })

            return results

        except Exception as e:
            print(f"⚠️ Spotify falló, usando fallback: {e}")

            emotion = params.get("dominant_emotion", "mixed")

            fallback_queries = {
                "joy": "happy upbeat party",
                "excitement": "high energy edm",
                "optimism": "positive pop indie",
                "amusement": "funk disco",
                "sadness": "sad acoustic chill",
                "grief": "piano ambient",
                "anger": "metal rock aggressive",
                "annoyance": "alternative rock",
                "fear": "calm relaxing",
                "neutral": "top hits",
                "disappointment": "sad slow chill"
            }

            query = fallback_queries.get(emotion, "top hits")

            try:
                return self.spotify.search_track(query, limit=n_results)
            except:
                return []

    # ====================================================================================
    # PLAYLISTS POR CONTEXTO
    # ====================================================================================
    def _get_context_based_playlists(self, contexts, dominant_emotion):

        mapping = {
            "study/work": f"{dominant_emotion} focus study",
            "party": f"{dominant_emotion} reggaeton party",
            "workout": f"{dominant_emotion} gym motivation",
            "relax/sleep": f"{dominant_emotion} relaxing calm",
            "driving": f"{dominant_emotion} driving roadtrip"
        }

        results = []

        for ctx in contexts[:2]:
            if ctx in mapping:
                try:
                    q = mapping[ctx]
                    search = self.spotify.sp.search(q=q, type="playlist", limit=2)
                    for item in search["playlists"]["items"]:
                        results.append({
                            "name": item["name"],
                            "url": item["external_urls"]["spotify"],
                            "context": ctx
                        })
                except:
                    pass

        return results

    # ====================================================================================
    # EXPLICACIÓN PARA LA UI
    # ====================================================================================
    def _generate_personalized_explanation(self, emotion_data, enriched_context):

        emo = emotion_data["dominant_emotion"]
        val = emotion_data["dimensions"]["valence"]
        ene = emotion_data["dimensions"]["energy"]
        genres = emotion_data["suggested_genres"][:3]

        text = "🎵 "

        prefs = enriched_context.get("music_preferences", {})
        total = prefs.get("total_interactions", 0)

        if total > 10:
            text += "Como ya te conozco bastante, "
        elif total > 3:
            text += "Recordando nuestras conversaciones, "
        else:
            text += "Basado en lo que me cuentas, "

        text += f"percibo que estás **{emo}**, "

        if ene < 0.3:
            text += "así que busqué música suave, "
        elif ene > 0.7:
            text += "así que busqué música muy energética, "
        else:
            text += "así que busqué música equilibrada, "

        text += f"con una valencia emocional de {val:.2f}. "

        if genres:
            text += f"Los géneros ideales para ti son: {', '.join(genres)}."

        return text
