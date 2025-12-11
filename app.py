"""
RhythmAI - Sistema Inteligente de Recomendación Musical.

Aplicación Streamlit que utiliza análisis emocional y vectorización semántica
para recomendar música personalizada basada en el estado de ánimo del usuario.
"""

import streamlit as st
import time
import traceback
import logging

from rhythmai.core.music_recommender import MusicRecommender

st.set_page_config(
    page_title="RhythmAI | Recomendación Musical Inteligente",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    :root {
        --primary: #00a650;
        --primary-dark: #008f44;
        --bg-dark: #0f172a;
        --bg-medium: #1e293b;
        --bg-light: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --border: #334155;
    }

    * {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }

    .main {
        background: #ffffff;
        padding: 2rem 1rem;
    }

    .block-container {
        max-width: 1200px;
        padding: 2rem 1.5rem;
    }

    .header-container {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 166, 80, 0.15);
    }

    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-dark) 0%, var(--bg-medium) 100%) !important;
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] strong {
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] .stMetric {
        background: var(--bg-medium) !important;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid var(--border);
        transition: all 0.3s ease;
        margin: 0.75rem 0;
    }

    section[data-testid="stSidebar"] .stMetric:hover {
        background: var(--bg-light) !important;
        border-color: var(--primary);
        transform: translateX(4px);
    }

    section[data-testid="stSidebar"] .stMetric label {
        color: var(--text-muted) !important;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700;
        font-size: 1.75rem;
    }

    section[data-testid="stSidebar"] .stProgress > div > div {
        background: var(--primary) !important;
    }

    section[data-testid="stSidebar"] .stProgress > div {
        background: var(--bg-dark) !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
        opacity: 0.4;
        margin: 1.5rem 0;
    }

    .stButton>button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.9rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 166, 80, 0.25);
        width: 100%;
    }

    .stButton>button:hover {
        background: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 166, 80, 0.35);
    }

    section[data-testid="stSidebar"] .stButton>button {
        background: var(--bg-light) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        font-size: 0.85rem;
        padding: 0.7rem 1.25rem;
    }

    section[data-testid="stSidebar"] .stButton>button:hover {
        background: var(--primary) !important;
        border-color: var(--primary) !important;
    }

    .stTextArea textarea {
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: #ffffff;
    }

    .stTextArea textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(0, 166, 80, 0.1);
    }

    .track-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        position: relative;
    }

    .track-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 4px;
        height: 100%;
        background: var(--primary);
        border-radius: 10px 0 0 10px;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .track-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
        border-color: var(--primary);
    }

    .track-card:hover::before {
        opacity: 1;
    }

    .stMetric {
        background: #f8fafc;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }

    .stMetric:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 12px rgba(0, 166, 80, 0.08);
    }

    .stMetric label {
        color: #64748b !important;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-weight: 700;
    }

    .stProgress > div > div {
        background: var(--primary) !important;
    }

    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: #e2e8f0;
    }

    .stSuccess {
        background-color: #f0fdf4;
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 1rem !important;
    }

    .stInfo {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem !important;
    }

    .stWarning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem !important;
    }

    img {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }

    .icon {
        margin-right: 0.4rem;
    }

    .stSpinner > div {
        border-top-color: var(--primary) !important;
    }

    a {
        color: var(--primary);
        text-decoration: none;
        transition: opacity 0.2s ease;
    }

    a:hover {
        opacity: 0.8;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: var(--text-muted) !important;
        font-size: 0.8rem;
    }

    .placeholder-album {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        width: 120px;
        height: 120px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_recommender():
    """
    Inicializa el sistema de recomendación musical.

    Returns:
        MusicRecommender: Instancia del sistema de recomendación.
    """
    try:
        with st.spinner("Inicializando sistema de IA..."):
            return MusicRecommender(user_id="streamlit_user")
    except Exception as e:
        st.error(f"Error al inicializar el sistema: {str(e)}")
        st.code(traceback.format_exc())
        return None


EMOTION_TRANSLATIONS = {
    "joy": "Alegría",
    "sadness": "Tristeza",
    "anger": "Enfado",
    "fear": "Miedo",
    "love": "Amor",
    "surprise": "Sorpresa",
    "neutral": "Neutral",
    "excitement": "Emoción/Energía",
    "focus": "Concentración",
    "sleep": "Sueño/Descanso",
    "party": "Fiesta/Celebración",
    "workout": "Entrenamiento",
    "chill": "Relajación",
    "grief": "Pena/Duelo",
    "disappointment": "Decepción",
    "optimism": "Optimismo",
    "approval": "Aprobación",
    "amusement": "Diversión",
    "annoyance": "Molestia",
    "disapproval": "Desaprobación",
    "remorse": "Remordimiento",
    "nostalgic": "Nostalgia",
    "motivated": "Motivación",
    "stressed": "Estrés",
    "confident": "Confianza",
    "relaxed": "Relajado",
    "bored": "Aburrimiento"
}


def translate_emotion(emotion):
    """Traduce una emoción del inglés al castellano."""
    return EMOTION_TRANSLATIONS.get(emotion.lower(), emotion.title())


recommender = load_recommender()

if recommender is None:
    st.stop()

st.markdown("""
<div class="header-container">
    <h1 class="header-title">
        <i class="fa-solid fa-headphones"></i>
        RhythmAI
    </h1>
    <p class="header-subtitle">Recomendación Musical Inteligente con Análisis Emocional</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3><i class='fa-solid fa-user icon'></i>Perfil Musical</h3>", unsafe_allow_html=True)
    st.markdown("---")

    try:
        prefs = recommender.context_manager.conversation_memory.get_music_preferences()
        recent_emotions = recommender.context_manager.conversation_memory.get_emotion_history(n=5)

        if prefs and prefs.get("total_interactions", 0) > 0:
            st.metric(
                "Sesiones Totales",
                prefs["total_interactions"]
            )

            st.markdown("---")

            if prefs.get("favorite_genres"):
                st.markdown("<h4><i class='fa-solid fa-music icon'></i>Géneros Principales</h4>", unsafe_allow_html=True)
                for idx, (genre, count) in enumerate(prefs["favorite_genres"][:5], 1):
                    percentage = (count / prefs["total_interactions"]) * 100
                    st.markdown(f"<p><strong>{idx}.</strong> {genre.title()}</p>", unsafe_allow_html=True)
                    st.caption(f"{count} sesiones ({percentage:.0f}%)")
                    st.progress(percentage / 100)
                    st.markdown("")

            st.markdown("---")

            if prefs.get("common_emotions"):
                st.markdown("<h4><i class='fa-solid fa-heart icon'></i>Emociones Frecuentes</h4>", unsafe_allow_html=True)
                for emo, count in prefs["common_emotions"][:5]:
                    st.markdown(f"<p><strong>{translate_emotion(emo)}</strong> - {count}x</p>", unsafe_allow_html=True)

        else:
            st.info("Comienza tu viaje musical describiendo cómo te sientes.")

        st.markdown("---")

        if recent_emotions:
            st.markdown("<h4><i class='fa-solid fa-chart-line icon'></i>Actividad Reciente</h4>", unsafe_allow_html=True)
            for entry in recent_emotions[-5:]:
                emo = entry.get("emotion", "desconocido")
                score = entry.get("score", 0)

                try:
                    score = float(score) if score else 0.0
                except:
                    score = 0.0

                st.caption(f"{translate_emotion(emo)} ({score:.0%})")

        st.markdown("---")

        if st.button("Reiniciar Datos", help="Borrar perfil e historial"):
            recommender.context_manager.clear_all()
            st.success("Perfil reiniciado")
            time.sleep(1)
            st.rerun()

    except Exception as e:
        st.warning("Datos de perfil no disponibles")
        logging.error(f"Error en sidebar: {str(e)}")

st.markdown("### <i class='fa-solid fa-comment-dots icon'></i>¿Cómo te sientes hoy?", unsafe_allow_html=True)

user_input = st.text_area(
    label="Tu estado emocional",
    placeholder="Describe tu estado de ánimo, nivel de energía o actividad actual...\nEjemplo: Me siento con energía y quiero música motivadora para entrenar",
    height=120,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_button = st.button("Obtener Recomendaciones", type="primary", use_container_width=True)

if recommend_button:
    if not user_input.strip():
        st.warning("Por favor describe primero cómo te sientes")
        st.stop()

    with st.spinner("Analizando emociones..."):
        progress_bar = st.progress(0)
        start_time = time.time()

        try:
            progress_bar.progress(20)
            results = recommender.recommend(user_input, n_results=10, randomize=True)
            progress_bar.progress(100)

            elapsed = time.time() - start_time
            st.success(f"Análisis completado en {elapsed:.2f}s")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()

    st.markdown("---")
    st.markdown("## <i class='fa-solid fa-brain icon'></i>Análisis Emocional", unsafe_allow_html=True)

    emotion_data = results["emotion_analysis"]

    dom = emotion_data.get("dominant_emotion", "neutral")
    dom_score = emotion_data.get("dominant_score", 0)

    try:
        dom_score = float(dom_score) if dom_score else 0.0
    except:
        dom_score = 0.0

    st.metric("Emoción Dominante", translate_emotion(dom), f"{dom_score:.0%} confianza")

    if emotion_data.get("suggested_genres"):
        genres = ", ".join([g.title() for g in emotion_data["suggested_genres"]])
        st.info(f"**Géneros Recomendados:** {genres}")

    st.markdown("---")
    st.markdown("## <i class='fa-solid fa-list-music icon'></i>Recomendaciones", unsafe_allow_html=True)

    tracks = results.get("vector_results", [])

    if tracks:
        st.info(f"Encontradas {len(tracks)} canciones perfectas para ti")

        for idx, t in enumerate(tracks, 1):
            with st.container():
                col1, col2 = st.columns([1, 5])

                with col1:
                    # Verificar si hay imagen de álbum y si es válida
                    album_image = t.get("album_image")
                    has_valid_image = album_image and isinstance(album_image, str) and album_image.strip() and album_image != "None"

                    if has_valid_image:
                        try:
                            st.image(album_image, width=120)
                        except:
                            # Si falla la carga de imagen, mostrar placeholder
                            st.markdown("<div class='placeholder-album'>🎵</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='placeholder-album'>🎵</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown(f"### {idx}. {t.get('name', 'Canción Desconocida')}")
                    st.markdown(f"**<i class='fa-solid fa-user icon'></i>Artista:** {t.get('artist', 'Artista Desconocido')}", unsafe_allow_html=True)

                    # Mostrar género si está disponible
                    genre = t.get("genre")
                    if genre and genre != "unknown":
                        st.markdown(f"**<i class='fa-solid fa-guitar icon'></i>Género:** {genre.title()}", unsafe_allow_html=True)

                    # Verificar preview URL
                    preview_url = t.get("preview_url")
                    has_valid_preview = preview_url and isinstance(preview_url, str) and preview_url.strip() and preview_url != "None"

                    if has_valid_preview:
                        try:
                            st.audio(preview_url, format="audio/mp3")
                        except Exception as audio_error:
                            st.caption("⚠️ Vista previa no disponible")
                            logging.warning(f"Error al cargar audio: {audio_error}")
                    else:
                        st.caption("⚠️ Vista previa no disponible")

                    # Añadir enlace a Deezer si está disponible
                    deezer_url = t.get("url")
                    if deezer_url and deezer_url != "#" and deezer_url != "None":
                        st.markdown(f"[🎧 Escuchar en Deezer]({deezer_url})", unsafe_allow_html=True)

                st.markdown("---")
    else:
        st.warning("No se encontraron recomendaciones. Intenta describir tu estado de otra manera.")

        # Sugerencia de debug
        if st.checkbox("Mostrar información de depuración"):
            st.json({
                "emotion_data": emotion_data,
                "db_count": recommender.vector_store.count() if recommender.vector_store else 0
            })

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1.5rem; color: #64748b;'>
    <p style='font-size: 1rem; font-weight: 500;'>RhythmAI - Recomendación Musical Inteligente</p>
    <p style='margin-top: 0.5rem; color: #94a3b8; font-size: 0.875rem;'>Impulsado por IA, Análisis Emocional y Bases de Datos Vectoriales</p>
</div>
""", unsafe_allow_html=True)