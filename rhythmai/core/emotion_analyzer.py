"""
Analizador de emociones con análisis semántico avanzado.
Utiliza embeddings para comprender el contexto sin depender de listas de palabras predefinidas.
"""

import logging
import re
import pickle
import hashlib
from pathlib import Path
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

from rhythmai.config import Config

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """
    Analizador de emociones con comprensión semántica avanzada.

    Combina tres técnicas de análisis:
    1. Modelo de análisis de sentimientos (positivo/negativo/neutral)
    2. Embeddings semánticos para clasificar tipo de actividad
    3. Clasificación basada en similitud semántica sin listas hardcodeadas
    """

    def __init__(self, embedder=None):
        """
        Inicializa los modelos de análisis emocional.

        Args:
            embedder: Instancia de SentenceTransformer para reutilizar.
                     Si es None, se crea una nueva instancia.

        Raises:
            RuntimeError: Si no se pueden cargar los modelos requeridos.
        """
        logger.info("Cargando modelos de análisis emocional...")

        try:
            # Inicializar modelo de análisis de sentimientos
            # Configurar device según Config.USE_GPU: -1 para CPU, 0 para GPU
            device = 0 if Config.USE_GPU else -1
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=Config.EMOTION_MODEL,
                device=device,
                top_k=None
            )

            # Reutilizar embedder existente o crear uno nuevo
            if embedder is not None:
                logger.info("Reutilizando instancia existente de embedder")
                self.embedder = embedder
            else:
                logger.info("Cargando nuevo modelo de embeddings semánticos...")
                self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)

            # Construir prototipos de actividades mediante embeddings
            self.activity_prototypes = self._build_prototypes({
                'high_energy': [
                    'bailar', 'fiesta', 'celebrar', 'entrenar', 'gimnasio',
                    'correr', 'ejercicio intenso', 'moverme', 'activarme'
                ],
                'low_energy': [
                    'estudiar', 'concentrarme', 'leer', 'trabajar',
                    'relajarme', 'descansar', 'tranquilidad'
                ],
                'happy': [
                    'feliz', 'alegre', 'contento', 'alegría', 'felicidad',
                    'animado', 'bien', 'genial', 'fantástico', 'dichoso'
                ],
                'romantic': [
                    'cita romántica', 'pareja', 'amor', 'romántico',
                    'momento íntimo', 'aniversario'
                ],
                'sad': [
                    'triste', 'llorar', 'melancolía', 'dolor',
                    'tristeza', 'pena', 'soledad'
                ],
                'angry': [
                    'rabia', 'enfado', 'frustración', 'ira',
                    'molesto', 'enojado', 'irritado'
                ],
                'sleep': [
                    'dormir', 'sueño', 'descanso nocturno',
                    'conciliar sueño', 'noche'
                ],
                'party': [
                    'fiesta', 'rumba', 'discoteca', 'salir de fiesta',
                    'celebración', 'pasarla bien'
                ],
                'workout': [
                    'gimnasio', 'gym', 'pesas', 'entrenar duro',
                    'rutina ejercicio', 'fitness'
                ],
                'nostalgic': [
                    'nostalgia', 'recuerdos', 'pasado', 'extrañar',
                    'tiempos antiguos', 'memorias', 'recordar'
                ],
                'motivated': [
                    'motivación', 'motivado', 'inspiración', 'inspirado',
                    'empujón', 'ánimo', 'impulso'
                ],
                'stressed': [
                    'estrés', 'estresado', 'agobio', 'presión',
                    'ansiedad', 'nervios', 'tensión'
                ],
                'confident': [
                    'confianza', 'seguro', 'empoderado', 'fuerte',
                    'capaz', 'poder', 'autoestima'
                ],
                'relaxed': [
                    'relajado', 'tranquilo', 'paz', 'calma',
                    'sereno', 'descanso', 'sosiego'
                ],
                'bored': [
                    'aburrido', 'aburrimiento', 'tedio', 'monotonía',
                    'sin hacer nada', 'rutina pesada'
                ]
            })

            logger.info("Modelos cargados exitosamente")

        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            raise RuntimeError(
                "No se pudieron cargar los modelos. "
                "Instala: pip install transformers torch sentencepiece sentence-transformers"
            )

    def _build_prototypes(self, keyword_groups):
        """
        Construye prototipos enriquecidos automáticamente a partir de palabras clave.

        Genera variaciones mediante plantillas de lenguaje natural para crear
        un prototipo semántico robusto sin necesidad de enumerar todas las variantes manualmente.

        Usa cache en disco para evitar regenerar prototipos en cada inicialización.

        Args:
            keyword_groups (dict): Diccionario de categoría a lista de palabras clave.

        Returns:
            dict: Prototipos representados como embeddings promediados.
        """
        # Generar hash único basado en keyword_groups y modelo de embeddings
        cache_key = self._generate_cache_key(keyword_groups)
        cache_dir = Path(Config.DATA_PATH) / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"prototypes_{cache_key}.pkl"

        # Intentar cargar desde cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    prototypes = pickle.load(f)
                logger.info(f"Prototipos cargados desde cache: {cache_file.name}")
                return prototypes
            except Exception as e:
                logger.warning(f"Error cargando cache, regenerando prototipos: {e}")

        # Si no hay cache, generar prototipos
        logger.info("Generando prototipos de emociones (esto puede tardar 5-10 segundos)...")
        prototypes = {}

        # Plantillas para generar variaciones automáticas de contexto
        templates = [
            "{keyword}",
            "música para {keyword}",
            "quiero {keyword}",
            "necesito {keyword}",
            "momento de {keyword}",
            "cuando estoy {keyword}",
            "para {keyword}",
            "mientras {keyword}",
            "estado de {keyword}",
            "sentirse {keyword}"
        ]

        for category, keywords in keyword_groups.items():
            variations = []

            # Generar todas las variaciones posibles para cada palabra clave
            for keyword in keywords:
                for template in templates:
                    variation = template.format(keyword=keyword)
                    variations.append(variation)

            # Calcular embedding promedio de todas las variaciones
            logger.info(f"Generando prototipo '{category}' con {len(variations)} variaciones")
            embeddings = self.embedder.encode(variations)
            prototype = embeddings.mean(axis=0)

            prototypes[category] = prototype

        # Guardar en cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(prototypes, f)
            logger.info(f"Prototipos guardados en cache: {cache_file.name}")
        except Exception as e:
            logger.warning(f"No se pudo guardar cache: {e}")

        return prototypes

    def _generate_cache_key(self, keyword_groups):
        """
        Genera un hash único para identificar los prototipos.

        Args:
            keyword_groups (dict): Diccionario de categoría a lista de palabras clave.

        Returns:
            str: Hash MD5 de 8 caracteres.
        """
        # Crear string determinista de keyword_groups y modelo
        content = f"{Config.EMBEDDING_MODEL}_{sorted(keyword_groups.items())}"
        hash_obj = hashlib.md5(content.encode())
        return hash_obj.hexdigest()[:8]

    def get_default_emotion_response(self, confidence=0.50):
        """
        Retorna una respuesta emocional por defecto (neutral).

        Método público para obtener una respuesta emocional neutral por defecto
        sin necesidad de acceder a métodos privados internos.

        Args:
            confidence (float): Nivel de confianza del análisis por defecto.

        Returns:
            dict: Respuesta emocional con estado neutral.
        """
        return self._build_emotion_response('neutral', confidence=confidence)

    def analyze(self, text):
        """
        Analiza el texto mediante comprensión semántica avanzada.

        Args:
            text (str): Texto del usuario a analizar.

        Returns:
            dict: Diccionario con análisis emocional completo incluyendo emoción dominante,
                  géneros sugeridos y parámetros de audio.
        """
        if not text or not text.strip():
            return self._build_emotion_response('neutral', confidence=0.50)

        try:
            text_clean = text.strip()[:512]

            # Fase 1: Análisis de sentimiento base
            results = self.sentiment_pipeline(text_clean)
            if isinstance(results[0], list):
                results = results[0]

            dominant = max(results, key=lambda x: x['score'])
            sentiment = dominant['label'].lower()
            sentiment_confidence = dominant['score']

            logger.info(f"Sentimiento: {sentiment} ({sentiment_confidence:.2%})")

            # Fase 2: Extracción del contexto o actividad del texto
            activity_context = self._extract_activity_context(text_clean)

            # Fase 3: Análisis semántico del contexto identificado
            if activity_context:
                logger.info(f"Analizando contexto: '{activity_context}'")
                emotion = self._analyze_semantic_context(
                    activity_context,
                    sentiment,
                    sentiment_confidence
                )
            else:
                # Analizar el texto completo si no se detectó actividad específica
                logger.info("No se detectó patrón específico, analizando texto completo")
                emotion = self._analyze_semantic_context(
                    text_clean,
                    sentiment,
                    sentiment_confidence
                )

            logger.info(f"Emoción final: {emotion}")
            return self._build_emotion_response(emotion, confidence=sentiment_confidence)

        except Exception as e:
            logger.error(f"Error en análisis: {e}")
            return self._build_emotion_response('neutral', confidence=0.50)

    def _extract_activity_context(self, text):
        """
        Extrae el contexto de actividad del texto mediante patrones lingüísticos.

        Identifica patrones como:
        - "para [actividad]"
        - "mientras [actividad]"
        - "quiero/necesito [actividad]"

        Args:
            text (str): Texto a analizar.

        Returns:
            str or None: Contexto de actividad identificado o None si no se detecta.
        """
        text_lower = text.lower()

        # Patrones de expresiones regulares para capturar intenciones
        patterns = [
            r'para\s+(\w+(?:\s+\w+){0,2})',
            r'mientras\s+(\w+(?:\s+\w+){0,2})',
            r'cuando\s+(\w+(?:\s+\w+){0,2})',
            r'al\s+(\w+(?:\s+\w+){0,2})',
            r'quiero\s+(\w+(?:\s+\w+){0,2})',
            r'necesito\s+(\w+(?:\s+\w+){0,2})',
            r'voy a\s+(\w+(?:\s+\w+){0,2})',
            r'(?:música|canciones)\s+(?:para|de)\s+(\w+(?:\s+\w+){0,2})',
        ]

        # Palabras a filtrar para evitar ruido en el contexto
        ignore_words = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'mi', 'tu', 'su', 'este', 'esta', 'ese', 'esa', 'mis',
            'tus', 'sus', 'mí', 'ti', 'música', 'canciones'
        }

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                activity = match.group(1).strip()

                # Filtrar palabras irrelevantes
                activity_words = activity.split()
                filtered_words = [w for w in activity_words if w not in ignore_words]

                if not filtered_words:
                    continue

                activity = ' '.join(filtered_words)

                # Obtener contexto ampliado alrededor de la actividad detectada
                words = text_lower.split()
                activity_first_word = activity.split()[0]

                if activity_first_word in words:
                    idx = words.index(activity_first_word)
                    context = ' '.join(words[max(0, idx-2):min(len(words), idx+4)])
                    logger.info(f"Contexto detectado: '{context}'")
                    return context

        return None

    def _analyze_semantic_context(self, context, sentiment, confidence):
        """
        Analiza el contexto mediante similitud semántica con prototipos predefinidos.

        Args:
            context (str): Contexto o texto a analizar.
            sentiment (str): Sentimiento base detectado.
            confidence (float): Nivel de confianza del análisis de sentimiento.

        Returns:
            str: Emoción identificada basada en similitud semántica.
        """
        try:
            # Generar embedding del contexto
            context_embedding = self.embedder.encode(context)

            # Calcular similitud coseno con cada prototipo
            similarities = {}
            for activity_type, prototype in self.activity_prototypes.items():
                similarity = util.cos_sim(context_embedding, prototype).item()
                similarities[activity_type] = similarity

            # Ordenar por similitud descendente
            sorted_similarities = sorted(
                similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            most_similar = sorted_similarities[0][0]
            max_similarity = sorted_similarities[0][1]

            # Registrar resultados de similitud
            logger.info("Similitudes semánticas (top 5):")
            for act_type, sim in sorted_similarities[:5]:
                logger.info(f"  - {act_type}: {sim:.3f}")
            logger.info(f"Más similar: {most_similar} ({max_similarity:.2%})")

            # Aplicar umbral adaptativo según confianza del sentimiento
            base_threshold = 0.35
            adjusted_threshold = base_threshold - (0.05 if confidence > 0.8 else 0)

            logger.info(f"Umbral usado: {adjusted_threshold:.2f}")

            if max_similarity > adjusted_threshold:
                result = self._activity_type_to_emotion(most_similar, sentiment)
                logger.info(f"Usando tipo de actividad: {most_similar} → {result}")
                return result
            else:
                logger.info(f"Similitud baja ({max_similarity:.2%}), usando solo sentimiento")
                return self._sentiment_to_emotion(sentiment)

        except Exception as e:
            logger.error(f"Error en análisis semántico: {e}")
            return self._sentiment_to_emotion(sentiment)

    def _activity_type_to_emotion(self, activity_type, sentiment):
        """
        Mapea el tipo de actividad detectado a una emoción específica.

        Considera el sentimiento base para desambiguar casos límite.

        Args:
            activity_type (str): Tipo de actividad identificada.
            sentiment (str): Sentimiento base.

        Returns:
            str: Emoción mapeada.
        """
        # Mapeo directo fuerte (alta confianza, sin ambigüedad)
        strong_mapping = {
            'happy': 'joy',
            'sad': 'sadness',
            'angry': 'anger',
            'romantic': 'love',
            'sleep': 'sleep',
            'workout': 'workout',
            'party': 'party',
            'nostalgic': 'nostalgic',
            'motivated': 'motivated',
            'stressed': 'stressed',
            'confident': 'confident',
            'relaxed': 'relaxed',
            'bored': 'bored'
        }

        # Si es mapeo fuerte, usarlo directamente
        if activity_type in strong_mapping:
            return strong_mapping[activity_type]

        # Mapeo débil (requiere contexto del sentimiento)
        weak_mapping = {
            'high_energy': {
                'positive': 'excitement',
                'negative': 'stressed',
                'neutral': 'excitement'
            },
            'low_energy': {
                'positive': 'relaxed',
                'negative': 'sadness',
                'neutral': 'focus'
            }
        }

        # Normalizar sentiment
        sentiment_normalized = sentiment
        if sentiment in ['pos']:
            sentiment_normalized = 'positive'
        elif sentiment in ['neg']:
            sentiment_normalized = 'negative'

        # Aplicar mapeo débil
        if activity_type in weak_mapping:
            mapping = weak_mapping[activity_type]
            if isinstance(mapping, dict):
                return mapping.get(sentiment_normalized, 'neutral')
            return mapping

        # Fallback al sentimiento base
        return self._sentiment_to_emotion(sentiment)

    def _sentiment_to_emotion(self, sentiment):
        """
        Mapea sentimiento básico a emoción genérica.

        Args:
            sentiment (str): Sentimiento base detectado.

        Returns:
            str: Emoción correspondiente.
        """
        mapping = {
            'positive': 'joy',
            'pos': 'joy',
            'negative': 'sadness',
            'neg': 'sadness',
            'neutral': 'neutral'
        }
        return mapping.get(sentiment, 'neutral')

    def _build_emotion_response(self, emotion, confidence=0.80):
        """
        Construye la respuesta completa del análisis emocional.

        Args:
            emotion (str): Emoción dominante detectada.
            confidence (float): Nivel de confianza del análisis.

        Returns:
            dict: Respuesta estructurada con emoción, géneros sugeridos, dimensiones
                  y parámetros de audio.
        """
        # Mapeo de emociones a géneros musicales
        emotion_to_genres = {
            # Emociones básicas
            'sadness': ['sad', 'chill', 'pop'],
            'joy': ['happy', 'pop', 'dance', 'party'],
            'anger': ['rock', 'workout'],
            'fear': ['chill', 'sad'],
            'love': ['pop', 'happy'],
            'neutral': ['pop', 'happy', 'party'],
            # Emociones específicas del sistema
            'excitement': ['party', 'dance', 'happy'],
            'focus': ['chill', 'pop'],
            'sleep': ['chill', 'sad'],
            'party': ['party', 'dance', 'happy'],
            'workout': ['workout', 'rock', 'party'],
            'chill': ['chill', 'sad', 'pop'],
            # Emociones extendidas
            'nostalgic': ['sad', 'pop', 'chill'],
            'motivated': ['workout', 'rock', 'party', 'happy'],
            'stressed': ['chill', 'sad'],
            'confident': ['pop', 'rock', 'party'],
            'relaxed': ['chill', 'pop'],
            'bored': ['pop', 'party', 'dance']
        }

        # Mapeo de emociones a dimensiones (valencia y energía)
        dimensions_map = {
            # Emociones básicas
            'sadness': {'valence': 0.2, 'energy': 0.3},
            'joy': {'valence': 0.9, 'energy': 0.7},
            'anger': {'valence': 0.3, 'energy': 0.9},
            'fear': {'valence': 0.3, 'energy': 0.4},
            'love': {'valence': 0.8, 'energy': 0.5},
            'neutral': {'valence': 0.5, 'energy': 0.5},
            # Emociones específicas del sistema
            'excitement': {'valence': 0.85, 'energy': 0.95},
            'focus': {'valence': 0.5, 'energy': 0.4},
            'sleep': {'valence': 0.6, 'energy': 0.15},
            'party': {'valence': 0.9, 'energy': 0.95},
            'workout': {'valence': 0.7, 'energy': 0.95},
            'chill': {'valence': 0.6, 'energy': 0.2},
            # Emociones extendidas
            'nostalgic': {'valence': 0.4, 'energy': 0.35},
            'motivated': {'valence': 0.8, 'energy': 0.85},
            'stressed': {'valence': 0.3, 'energy': 0.6},
            'confident': {'valence': 0.85, 'energy': 0.75},
            'relaxed': {'valence': 0.7, 'energy': 0.25},
            'bored': {'valence': 0.4, 'energy': 0.3}
        }

        dimensions = dimensions_map.get(emotion, {'valence': 0.5, 'energy': 0.5})

        return {
            'dominant_emotion': emotion,
            'dominant_score': confidence,
            'suggested_genres': emotion_to_genres.get(emotion, ['pop']),
            'dimensions': dimensions,
            'music_params': {
                'target_valence': dimensions['valence'],
                'target_energy': dimensions['energy']
            }
        }