import streamlit as st
import sys
from pathlib import Path
import time

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.music_recommender import MusicRecommender

st.set_page_config(page_title="RhythmAI", page_icon="🎧", layout="wide")

# CSS personalizado
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .emotion-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
    .metric-box {
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        background-color: #f0f2f6;
    }
    .memory-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #e8f4f8;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        width: 100%;
    }
    .track-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        transition: transform 0.2s;
    }
    .track-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Inicializar el recomendador con memoria
@st.cache_resource
def load_recommender():
    try:
        return MusicRecommender(user_id="streamlit_user")
    except Exception as e:
        st.error(f"Error al inicializar el recomendador: {e}")
        return None


recommender = load_recommender()

if recommender is None:
    st.stop()

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🎧 RhythmAI - DJ Conversacional Inteligente")
    st.markdown("*Sistema de IA que entiende tus emociones y recuerda tus preferencias musicales*")

# Sidebar con historial y perfil
with st.sidebar:
    st.header("🧠 Tu Perfil Musical")

    try:
        # Obtener contexto
        context = recommender.context_manager.get_enriched_context()

        # Mostrar estadísticas
        if context['music_preferences'] and context['music_preferences']['total_interactions'] > 0:
            prefs = context['music_preferences']

            st.metric("🎵 Conversaciones totales", prefs.get('total_interactions', 0))

            st.divider()

            # Géneros favoritos
            if prefs.get('favorite_genres'):
                st.subheader("🎸 Géneros favoritos")
                for genre, count in prefs['favorite_genres'][:5]:
                    percentage = (count / prefs['total_interactions']) * 100
                    st.write(f"• **{genre}** ({count}x - {percentage:.0f}%)")

            st.divider()

            # Emociones frecuentes
            if prefs.get('common_emotions'):
                st.subheader("😊 Emociones frecuentes")
                for emotion, count in prefs['common_emotions'][:5]:
                    st.write(f"• {emotion.capitalize()} ({count}x)")
        else:
            st.info(
                "👋 ¡Bienvenido! Aún no tengo información sobre tus preferencias. Empieza a conversar conmigo para que pueda conocerte mejor.")

        st.divider()

        # Historial emocional reciente
        if context.get('emotion_history') and len(context['emotion_history']) > 0:
            st.subheader("📊 Historial Emocional Reciente")
            for entry in context['emotion_history'][-5:]:
                emoji_map = {
                    'joy': '😊', 'sadness': '😢', 'anger': '😠',
                    'fear': '😨', 'surprise': '😲', 'love': '❤️',
                    'excitement': '🤩', 'neutral': '😐'
                }
                emoji = emoji_map.get(entry['emotion'], '🎭')
                st.caption(f"{emoji} {entry['emotion'].capitalize()} ({entry['score']:.0%})")

        st.divider()

        # Botón para limpiar memoria
        if st.button("🔄 Resetear memoria", type="secondary", help="Borra todo el historial y preferencias"):
            recommender.context_manager.clear_all()
            st.success("✅ Memoria limpiada correctamente")
            st.rerun()

    except Exception as e:
        st.error(f"Error al cargar perfil: {e}")

    st.divider()
    st.caption("💡 **Tip:** Cuanto más conversemos, mejores serán mis recomendaciones")

# Contenido principal
st.header("💬 Cuéntame cómo te sientes")

# Input principal
user_input = st.text_area(
    "Input",  # ← Añade un label
    placeholder="Ejemplo: 'Hoy estoy más animado que ayer, tengo ganas de hacer ejercicio pero también algo de nostalgia...'\n\nSé específico sobre tu estado de ánimo, lo que estás haciendo, o el ambiente que buscas.",
    height=150,
    key="user_input",
    label_visibility="hidden"  # ← Oculta el label visualmente
)

# Botón de recomendación
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_button = st.button("🎵 Recomiéndame música perfecta", type="primary", use_container_width=True)

if recommend_button:
    if user_input and user_input.strip():
        with st.spinner("🧠 Analizando tu estado emocional en detalle..."):
            start_time = time.time()
            try:
                results = recommender.recommend(user_input, n_results=8)
                processing_time = time.time() - start_time

                st.success(f"✅ Análisis completado en {processing_time:.2f} segundos")

                # ====================
                # SECCIÓN 1: RESUMEN Y EXPLICACIÓN
                # ====================
                st.markdown("---")
                st.markdown(f"### 🎵 {results['explanation']}")

                # ====================
                # SECCIÓN 2: ANÁLISIS EMOCIONAL DETALLADO
                # ====================
                st.markdown("---")
                st.header("🎭 Análisis Emocional Completo")

                emotion_data = results['emotion_analysis']

                # Resumen interpretable
                st.markdown(f"#### {emotion_data['summary']}")

                st.markdown("")

                # Métricas principales en 5 columnas
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric(
                        "🎯 Emoción Dominante",
                        emotion_data['dominant_emotion'].capitalize(),
                        f"{emotion_data['dominant_score']:.0%}"
                    )
                with col2:
                    energy_val = emotion_data['dimensions']['energy']
                    energy_label = "Alta" if energy_val > 0.6 else ("Media" if energy_val > 0.3 else "Baja")
                    st.metric("⚡ Energía", f"{energy_val:.2f}", energy_label)
                with col3:
                    valence_val = emotion_data['dimensions']['valence']
                    valence_label = "Positiva" if valence_val > 0.5 else "Negativa"
                    st.metric("😊 Valencia", f"{valence_val:.2f}", valence_label)
                with col4:
                    st.metric("🎚️ Intensidad", f"{emotion_data['dimensions']['intensity']:.2f}")
                with col5:
                    st.metric("🌀 Complejidad", f"{emotion_data['dimensions']['complexity']:.2f}")

                st.markdown("")

                # Top 5 emociones detectadas
                st.subheader("📊 Top 5 Emociones Detectadas")
                cols = st.columns(5)
                for idx, (emotion, score) in enumerate(emotion_data['top_emotions']):
                    with cols[idx]:
                        st.metric(f"#{idx + 1}", emotion.capitalize(), f"{score:.0%}")

                # Contexto detectado
                if emotion_data.get('context') and emotion_data['context'] != ['general']:
                    contexts_str = ', '.join([c.replace('_', ' ').title() for c in emotion_data['context']])
                    st.info(f"🎯 **Contexto detectado:** {contexts_str}")

                # Géneros sugeridos
                if emotion_data.get('suggested_genres'):
                    genres_str = ', '.join(emotion_data['suggested_genres'][:5])
                    st.success(f"🎸 **Géneros ideales para ti:** {genres_str}")

                # Parámetros técnicos (colapsable)
                with st.expander("🎛️ Parámetros Técnicos de Spotify", expanded=False):
                    params = emotion_data['spotify_params']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**🔋 Energy:** {params['target_energy']:.2f}")
                        st.write(f"**😊 Valence:** {params['target_valence']:.2f}")
                        st.write(f"**💃 Danceability:** {params['target_danceability']:.2f}")
                    with col2:
                        st.write(f"**🎸 Acousticness:** {params['target_acousticness']:.2f}")
                        st.write(f"**🎹 Instrumentalness:** {params['target_instrumentalness']:.2f}")
                        st.write(f"**⏱️ Tempo:** {params['tempo_range'][0]}-{params['tempo_range'][1]} BPM")

                # ====================
                # SECCIÓN 3: RECOMENDACIONES DE MÚSICA
                # ====================
                st.markdown("---")
                st.header("🎵 Tus Recomendaciones Personalizadas")

                if results['spotify_recommendations']:
                    for idx, track in enumerate(results['spotify_recommendations'], 1):
                        with st.container():
                            col1, col2 = st.columns([1, 5])

                            with col1:
                                if track.get('album_image'):
                                    st.image(track['album_image'], width=120)
                                else:
                                    st.write("🎵")

                            with col2:
                                st.markdown(f"### {idx}. {track['name']}")
                                st.markdown(f"**Artista:** {track['artist']}")
                                st.markdown(f"[▶️ Escuchar en Spotify]({track['url']})")

                                # Preview de audio si está disponible
                                if track.get('preview_url'):
                                    st.audio(track['preview_url'])

                            st.divider()
                else:
                    st.warning(
                        "⚠️ No se encontraron recomendaciones de Spotify. Intenta describir tu estado de ánimo de otra manera.")

                # ====================
                # SECCIÓN 4: PLAYLISTS POR CONTEXTO
                # ====================
                if results.get('context_playlists') and len(results['context_playlists']) > 0:
                    st.markdown("---")
                    st.subheader("🎧 Playlists Recomendadas por Contexto")
                    for playlist in results['context_playlists']:
                        context_emoji = {
                            'study/work': '📚',
                            'workout': '💪',
                            'relax/sleep': '😴',
                            'party': '🎉',
                            'driving': '🚗',
                            'emotional_release': '😢',
                            'morning': '🌅',
                            'night': '🌙'
                        }
                        emoji = context_emoji.get(playlist['context'], '🎵')
                        st.markdown(
                            f"{emoji} **{playlist['name']}** - *{playlist['context'].replace('_', ' ').title()}*")
                        st.markdown(f"[🔗 Ver playlist en Spotify]({playlist['url']})")
                        st.divider()

                # ====================
                # SECCIÓN 5: CONTEXTO DE MEMORIA (colapsable)
                # ====================
                if results.get('enriched_context'):
                    st.markdown("---")
                    with st.expander("🧠 Contexto de Memoria y Aprendizaje", expanded=False):
                        enriched = results['enriched_context']

                        # Conversaciones previas
                        if enriched['conversation_context'] != "Esta es tu primera conversación.":
                            st.markdown("### 💬 Conversaciones Previas")
                            st.text_area("", enriched['conversation_context'], height=200, disabled=True)

                        # Preferencias musicales
                        if enriched.get('music_preferences') and enriched['music_preferences'][
                            'total_interactions'] > 0:
                            st.markdown("### 🎵 Análisis de Preferencias")
                            prefs = enriched['music_preferences']

                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Géneros favoritos:**")
                                for genre, count in prefs['favorite_genres'][:5]:
                                    st.write(f"• {genre} ({count}x)")

                            with col2:
                                st.write("**Emociones comunes:**")
                                for emotion, count in prefs['common_emotions'][:5]:
                                    st.write(f"• {emotion} ({count}x)")

            except Exception as e:
                st.error(f"❌ Error al procesar la recomendación: {e}")
                import traceback

                with st.expander("Ver detalles del error"):
                    st.code(traceback.format_exc())

    else:
        st.warning("⚠️ Por favor, escribe cómo te sientes para poder recomendarte música")

# ====================
# SECCIÓN 6: EJEMPLOS DE USO
# ====================
st.markdown("---")
with st.expander("💡 Ejemplos de cómo describir tu estado de ánimo", expanded=False):
    st.markdown("""
    ### Ejemplos efectivos:

    ✅ **Detallado y específico:**
    - "Me siento con energía pero un poco ansioso porque tengo una reunión importante. Necesito música que me tranquilice pero que me mantenga alerta."

    ✅ **Con contexto:**
    - "Estoy estudiando para un examen y necesito concentrarme. Me siento un poco abrumado pero motivado."

    ✅ **Emocional:**
    - "Hoy ha sido un día difícil, estoy triste y necesito música que me acompañe pero que no me hunda más."

    ✅ **Actividad específica:**
    - "Voy a salir a correr por la mañana, me siento con energía y optimista, quiero música motivadora."

    ❌ **Poco efectivo:**
    - "Música"
    - "Algo bueno"
    - "No sé"
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🧠 Powered by RoBERTa GoEmotions (28 emociones)")
with col2:
    st.caption("🎵 Integrado con Spotify API")
with col3:
    st.caption("💾 Sistema de memoria con LangChain")

st.caption("💡 **Nota:** RhythmAI aprende de cada conversación para mejorar sus recomendaciones con el tiempo.")