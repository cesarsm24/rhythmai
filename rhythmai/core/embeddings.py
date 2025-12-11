"""
Modelo de embeddings para vectorización de texto

Utiliza Sentence-Transformers de HuggingFace para convertir
texto en vectores semánticos de alta dimensión.

Genera embeddings de 384 dimensiones optimizados para búsqueda
semántica y análisis de similitud textual.
"""

from sentence_transformers import SentenceTransformer
import logging
import numpy as np

from rhythmai.config import Config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Modelo de embeddings para conversión de texto a vectores semánticos.

    Utiliza Sentence-Transformers con el modelo all-MiniLM-L6-v2:
    - 384 dimensiones
    - Soporte multiidioma
    - Optimizado para similitud semántica
    - Tamaño aproximado: 80 MB
    """

    def __init__(self):
        """
        Inicializa el modelo de embeddings.

        Raises:
            Exception: Si falla la carga del modelo.
        """
        try:
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.dimensions = self.model.get_sentence_embedding_dimension()

            logger.info(f"Modelo de embeddings cargado: {Config.EMBEDDING_MODEL}")
            logger.info(f"Dimensiones del vector: {self.dimensions}")

        except Exception as e:
            logger.error(f"Error al cargar modelo de embeddings: {e}")
            raise

    def encode(self, text):
        """
        Convierte un texto en vector de embeddings.

        Args:
            text (str): Texto a vectorizar.

        Returns:
            numpy.ndarray: Vector de embeddings de dimensión configurada.

        Raises:
            ValueError: Si el texto es inválido o vacío.
        """
        if not text or not isinstance(text, str):
            raise ValueError("El texto debe ser una cadena no vacía")

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding

        except Exception as e:
            logger.error(f"Error al codificar texto: {e}")
            raise

    def encode_batch(self, texts):
        """
        Convierte múltiples textos en vectores de embeddings.

        Más eficiente que llamar encode() múltiples veces debido al
        procesamiento por lotes optimizado del modelo.

        Args:
            texts (list): Lista de textos a vectorizar.

        Returns:
            numpy.ndarray: Array de vectores de embeddings (n_texts × dimensiones).

        Raises:
            ValueError: Si la lista está vacía o contiene elementos inválidos.
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("texts debe ser una lista no vacía")

        valid_texts = [t for t in texts if t and isinstance(t, str)]

        if not valid_texts:
            raise ValueError("No hay textos válidos para codificar")

        try:
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )

            logger.info(
                f"Batch codificado: {len(valid_texts)} textos procesados, "
                f"dimensión resultante: {embeddings.shape[1]}"
            )

            return embeddings

        except Exception as e:
            logger.error(f"Error al codificar batch de textos: {e}")
            raise

    def get_similarity(self, text1, text2):
        """
        Calcula la similitud coseno entre dos textos.

        La similitud coseno mide el ángulo entre dos vectores,
        produciendo un valor entre 0 (completamente diferentes) y
        1 (semánticamente idénticos).

        Args:
            text1 (str): Primer texto.
            text2 (str): Segundo texto.

        Returns:
            float: Similitud coseno en el rango [0, 1].

        Raises:
            Exception: Si falla el cálculo de similitud.
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            embedding1 = self.encode(text1).reshape(1, -1)
            embedding2 = self.encode(text2).reshape(1, -1)

            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)

        except Exception as e:
            logger.error(f"Error al calcular similitud: {e}")
            raise

    def get_model_info(self):
        """
        Retorna información sobre el modelo cargado.

        Returns:
            dict: Diccionario con información del modelo incluyendo:
                - model_name: Identificador del modelo
                - dimensions: Dimensionalidad de los vectores
                - max_seq_length: Longitud máxima de secuencia soportada
                - device: Dispositivo de cómputo utilizado (CPU/GPU)
        """
        return {
            'model_name': Config.EMBEDDING_MODEL,
            'dimensions': self.dimensions,
            'max_seq_length': self.model.max_seq_length,
            'device': str(self.model.device)
        }