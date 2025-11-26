from transformers import pipeline
import numpy as np
import sys
from pathlib import Path

# Añadir directorio raíz
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config


class EmotionAnalyzer:
    """
    Analizador emocional optimizado para español.
    Ajusta emociones del modelo GoEmotions y evita falsos NEUTRAL.
    """

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=Config.EMOTION_MODEL,
            top_k=None
        )

        print(f"✅ Modelo de análisis emocional cargado: {Config.EMOTION_MODEL}")

        # Palabras clave según emoción
        self.positive_words = [
            "contento", "feliz", "alegre", "bien", "genial", "emocionado",
            "aprobé", "celebro", "me encanta", "fascinado", "ilusionado"
        ]

        self.negative_words = [
            "triste", "fatal", "deprimido", "mal", "horrible", "terrible",
            "ansioso", "nervioso", "estresado", "preocupado", "llorando",
            "perdí", "mal día", "solo", "vacío"
        ]

        self.anger_words = [
            "cabreado", "enfadado", "molesto", "harto", "rabia", "furioso"
        ]

    # ======================================================================
    # ========================== ANALIZAR EMOCIÓN ===========================
    # ======================================================================

    def analyze(self, text):

        text_low = text.lower()

        # Emociones brutas del modelo
        raw = self.classifier(text[:512])
        results = sorted(raw[0], key=lambda x: x["score"], reverse=True)

        # ==========================================================
        # 1) Subir PESO emociones positivas si aparecen palabras clave
        # ==========================================================
        if any(w in text_low for w in self.positive_words):
            for emo in results:
                if emo["label"] in ["joy", "excitement", "optimism", "approval", "love"]:
                    emo["score"] += 0.40

        # ==========================================================
        # 2) Subir PESO emociones negativas
        # ==========================================================
        if any(w in text_low for w in self.negative_words):
            for emo in results:
                if emo["label"] in ["sadness", "grief", "disappointment", "remorse", "fear"]:
                    emo["score"] += 0.45

        # ==========================================================
        # 3) Subir PESO emociones de enojo
        # ==========================================================
        if any(w in text_low for w in self.anger_words):
            for emo in results:
                if emo["label"] in ["anger", "annoyance", "disapproval"]:
                    emo["score"] += 0.50

        # ==========================================================
        # 4) Penalizar "neutral"
        # ==========================================================
        if results[0]["label"] == "neutral":
            if (
                any(w in text_low for w in self.positive_words) or
                any(w in text_low for w in self.negative_words) or
                any(w in text_low for w in self.anger_words)
            ):
                results[0]["score"] *= 0.25  # reducir el neutral

        # Reordenar de nuevo tras ajustes
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # Top 5 emociones (darán igual si son dicts o tuples)
        top = results[:5]

        # Dimensiones
        dimensions = self._calculate_dimensions(results)

        # Parámetros Spotify
        spotify_params = self._spotify_params(dimensions)

        # Géneros sugeridos
        genres = self._suggest_genres(top, dimensions)

        # Contexto
        context = self._detect_context(text_low)

        return {
            "emotions": {e["label"]: e["score"] for e in results},
            "top_emotions": top,
            "dominant_emotion": top[0]["label"],
            "dominant_score": top[0]["score"],
            "dimensions": dimensions,
            "spotify_params": spotify_params,
            "suggested_genres": genres,
            "context": context
        }

    # ======================================================================
    # ======================== DIMENSIONES EMOCIONALES ======================
    # ======================================================================

    def _calculate_dimensions(self, emotions):

        def get(label):
            return next((e["score"] for e in emotions if e["label"] == label), 0)

        # Valencia
        positive = get("joy") + get("amusement") + get("excitement") + get("love") + get("optimism") + get("approval")
        negative = get("sadness") + get("grief") + get("anger") + get("disappointment") + get("fear") + get("remorse")

        valence = (positive - negative + 1) / 2
        valence = max(0, min(1, valence))

        # Energía
        energy = (
            get("excitement")
            + get("anger") * 0.9
            + get("surprise") * 0.7
            - get("sadness") * 0.6
            - get("neutral") * 0.2
        )

        energy = (energy + 1) / 2
        energy = max(0, min(1, energy))

        return {
            "valence": round(valence, 3),
            "energy": round(energy, 3)
        }


    # ======================================================================
    # ====================== PARÁMETROS PARA SPOTIFY =======================
    # ======================================================================

    def _spotify_params(self, dim):

        if dim["energy"] > 0.7:
            tempo = (120, 150)
        elif dim["energy"] > 0.4:
            tempo = (90, 120)
        else:
            tempo = (60, 90)

        return {
            "target_energy": dim["energy"],
            "target_valence": dim["valence"],
            "tempo_range": tempo
        }

    # ======================================================================
    # ========================= SUGERENCIA DE GÉNEROS ======================
    # ======================================================================

    def _suggest_genres(self, top, dim):

        emotion = top[0]["label"]

        mapping = {
            "joy": ["dance", "pop", "party", "reggaeton"],
            "excitement": ["edm", "electronic", "techno"],
            "optimism": ["indie-pop", "pop"],
            "amusement": ["funk", "disco", "indie"],
            "sadness": ["acoustic", "indie", "piano", "sad"],
            "grief": ["ambient", "piano"],
            "anger": ["rock", "metal", "hardcore"],
            "annoyance": ["alternative", "grunge"],
            "fear": ["ambient", "lofi"],
            "disappointment": ["slow", "sad"],
            "neutral": ["pop", "indie"],
        }

        return mapping.get(emotion, ["pop"])

    # ======================================================================
    # ========================== DETECTAR CONTEXTO =========================
    # ======================================================================

    def _detect_context(self, text):
        contexts = []

        if "estudio" in text or "estudiar" in text or "trabajo" in text:
            contexts.append("study/work")
        if "gym" in text or "entreno" in text or "correr" in text:
            contexts.append("workout")
        if "relax" in text or "dormir" in text or "calma" in text:
            contexts.append("relax/sleep")
        if "fiesta" in text or "bailar" in text or "discoteca" in text:
            contexts.append("party")

        return contexts if contexts else ["general"]
