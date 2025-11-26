import streamlit as st
import sys
from pathlib import Path
import time

# Añadir directorio raíz
sys.path.insert(0, str(Path(__file__).parent))

from src.core.music_recommender import MusicRecommender


st.set_page_config(page_title="RhythmAI", page_icon="🎧", layout="wide")

# ===========================
# 🔧 Estilos CSS
# ===========================
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .track-card {
        padding: 15px; border-radius: 10px;
        background-color: #fff; border: 1px solid #e0e0e0;
        margin: 10px 0; transition: 0.2s;
    }
    .track-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ===========================
# 🔥 Cargar Recommender
# ===========================
@st.cache_resource
def load_recommender():
    try:
        return MusicRecommender(user_id="streamlit_user")
    except Exception as e:
        st.error(f"Error al inicializar: {e}")
        return None


recommender = load_recommender()

if recommender is None:
    st.stop()


# ===========================
# HEADER
# ===========================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🎧 RhythmAI - DJ Conversacional Inteligente")
    st.markdown("*Tu recomendador musical impulsado por IA y emociones.*")


# ===========================
# SIDEBAR (Perfil del usuario)
# ===========================
with st.sidebar:
    st.header("🧠 Tu Perfil Musical")

    try:
        context = recommender.context_manager.get_enriched_context()

        prefs = context.get("music_preferences", {})
        emotion_hist = context.get("emotion_history", [])

        # ----- ESTADÍSTICAS -----
        if prefs and prefs.get("total_interactions", 0) > 0:
            st.metric("🎵 Conversaciones totales", prefs["total_interactions"])

            # Géneros favoritos
            if prefs.get("favorite_genres"):
                st.subheader("🎸 Géneros favoritos")
                for genre, c in prefs["favorite_genres"][:5]:
                    st.write(f"- **{genre}** ({c} veces)")

            st.divider()

            # Emociones frecuentes
            if prefs.get("common_emotions"):
                st.subheader("😊 Emociones frecuentes")
                for emo, c in prefs["common_emotions"][:5]:
                    st.write(f"- {emo.capitalize()} ({c} veces)")

        else:
            st.info("👋 Aún no tengo información sobre ti. ¡Empieza a hablar conmigo!")

        st.divider()

        # Historial emocional
        if emotion_hist:
            st.subheader("📊 Historial Reciente")
            for entry in emotion_hist[-5:]:

                emo = entry.get("emotion", "desconocido")
                score = entry.get("score", 0)

                try:
                    score = float(score)
                except:
                    score = 0.0

                st.caption(f"- {emo.capitalize()} ({score:.0%})")

        st.divider()

        if st.button("🗑️ Resetear memoria"):
            recommender.context_manager.clear_all()
            st.success("Memoria eliminada")
            st.rerun()

    except Exception as e:
        st.error(f"Error en sidebar: {e}")


# ===========================
# INPUT PRINCIPAL
# ===========================
st.header("💬 Cuéntame cómo te sientes")

user_input = st.text_area(
    "Describe tu estado emocional",
    placeholder="Ejemplo: Estoy muy feliz, tengo ganas de bailar...",
    height=150
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_button = st.button("🎵 Recomiéndame música", type="primary", use_container_width=True)


# ===========================
# PROCESAR RECOMENDACIÓN
# ===========================
if recommend_button:
    if not user_input.strip():
        st.warning("⚠️ Por favor, escribe cómo te sientes.")
        st.stop()

    with st.spinner("🧠 Analizando emociones..."):
        start = time.time()

        try:
            results = recommender.recommend(user_input, n_results=8)
        except Exception as e:
            st.error(f"❌ Error al procesar la recomendación: {e}")
            st.stop()

        st.success(f"Análisis completo en {time.time() - start:.2f}s")


        # ===========================
        # EXPLICACIÓN PERSONALIZADA
        # ===========================
        st.markdown("---")
        st.markdown(f"### 🎵 {results['explanation']}")


        # ===========================
        # ANÁLISIS EMOCIONAL
        # ===========================
        st.markdown("---")
        st.header("🎭 Análisis Emocional Completo")

        emotion_data = results["emotion_analysis"]

        # ---- MÉTRICA PRINCIPAL ----
        dom = emotion_data.get("dominant_emotion", "neutral")
        dom_score = emotion_data.get("dominant_score", 0)

        try:
            dom_score = float(dom_score)
        except:
            dom_score = 0.0

        st.metric("🎯 Emoción Dominante", dom.capitalize(), f"{dom_score:.0%}")

        # ---- DIMENSIONES ----
        dims = emotion_data.get("dimensions", {})

        col1, col2 = st.columns(2)
        with col1:
            st.metric("⚡ Energía", f"{dims.get('energy', 0):.2f}")
        with col2:
            st.metric("😊 Valencia", f"{dims.get('valence', 0):.2f}")


        # ---- TOP 5 EMOCIONES ----
        st.subheader("📊 Top 5 Emociones Detectadas")

        cols = st.columns(5)
        for idx, item in enumerate(emotion_data.get("top_emotions", [])[:5]):

            if isinstance(item, dict):
                emo = item.get("label", "desconocida")
                score = item.get("score", 0)
            else:
                emo, score = item

            try:
                score = float(score)
            except:
                score = 0.0

            with cols[idx]:
                st.metric(f"#{idx+1}", emo.capitalize(), f"{score:.0%}")


        # ---- Géneros sugeridos ----
        if emotion_data.get("suggested_genres"):
            genres = ", ".join(emotion_data["suggested_genres"])
            st.success(f"🎸 Géneros ideales: {genres}")


        # ===========================
        # RECOMENDACIONES DE SPOTIFY
        # ===========================
        st.markdown("---")
        st.header("🎵 Recomendaciones de Spotify")

        tracks = results["spotify_recommendations"]

        if tracks:
            for idx, t in enumerate(tracks, 1):
                with st.container():
                    c1, c2 = st.columns([1, 5])

                    with c1:
                        if t.get("album_image"):
                            st.image(t["album_image"], width=120)
                        else:
                            st.write("🎵")

                    with c2:
                        st.markdown(f"### {idx}. {t['name']}")
                        st.markdown(f"**Artista:** {t['artist']}")
                        st.markdown(f"[▶️ Escuchar en Spotify]({t['url']})")

                        if t.get("preview_url"):
                            st.audio(t["preview_url"])

                    st.divider()
        else:
            st.warning("⚠️ No se encontraron recomendaciones.")


        # ===========================
        # PLAYLISTS DE CONTEXTO
        # ===========================
        playlists = results.get("context_playlists", [])
        if playlists:
            st.markdown("---")
            st.subheader("🎧 Playlists según tu contexto")

            for p in playlists:
                st.markdown(f"**{p['name']}** — {p['context'].title()}")
                st.markdown(f"[🔗 Ver en Spotify]({p['url']})")
                st.divider()


        # ===========================
        # MEMORIA CONTEXTUAL
        # ===========================
        enriched = results.get("enriched_context")
        if enriched:
            st.markdown("---")
            with st.expander("🧠 Memoria y Aprendizaje"):
                st.json(enriched)


# ===========================
# EJEMPLOS
# ===========================
st.markdown("---")
with st.expander("💡 Ejemplos de frases efectivas"):
    st.markdown("""
### Ejemplos:
- "Estoy cansado pero quiero algo motivador."
- "Estoy triste y necesito música suave."
- "Hoy estoy muy feliz, quiero fiesta."
- "Estoy estudiando y necesito concentración."
    """)
