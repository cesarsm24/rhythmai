from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config


class EmbeddingModel:
    """
    Modelo de embeddings para convertir texto en vectores semánticos
    Usa Sentence Transformers para búsqueda semántica
    """

    def __init__(self):
        try:
            # Modelo ligero y eficiente (383 MB, muy rápido)
            # 384 dimensiones, multiidioma, optimizado para similitud semántica
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
            print(f"✅ Modelo de embeddings cargado: {Config.EMBEDDING_MODEL}")
        except Exception as e:
            print(f"❌ Error al cargar modelo de embeddings: {e}")
            raise

    def encode(self, text):
        """
        Convierte un texto en vector de embeddings

        Args:
            text (str): Texto a convertir

        Returns:
            numpy.ndarray: Vector de embeddings (384 dimensiones)
        """
        try:
            if not text or not isinstance(text, str):
                raise ValueError("El texto debe ser una cadena no vacía")

            # Encode retorna un numpy array
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding

        except Exception as e:
            print(f"❌ Error al codificar texto: {e}")
            raise

    def encode_batch(self, texts):
        """
        Convierte múltiples textos en vectores de embeddings
        Más eficiente que llamar encode() múltiples veces

        Args:
            texts (list): Lista de textos a convertir

        Returns:
            numpy.ndarray: Array de vectores de embeddings
        """
        try:
            if not texts or not isinstance(texts, list):
                raise ValueError("texts debe ser una lista no vacía")

            # Filtrar textos vacíos
            valid_texts = [t for t in texts if t and isinstance(t, str)]

            if not valid_texts:
                raise ValueError("No hay textos válidos para codificar")

            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )

            return embeddings

        except Exception as e:
            print(f"❌ Error al codificar batch de textos: {e}")
            raise

    def get_similarity(self, text1, text2):
        """
        Calcula la similitud coseno entre dos textos

        Args:
            text1 (str): Primer texto
            text2 (str): Segundo texto

        Returns:
            float: Similitud entre 0 y 1 (1 = idénticos)
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            embedding1 = self.encode(text1).reshape(1, -1)
            embedding2 = self.encode(text2).reshape(1, -1)

            similarity = cosine_similarity(embedding1, embedding2)[0][0]

            return float(similarity)

        except Exception as e:
            print(f"❌ Error al calcular similitud: {e}")
            raise

    def get_model_info(self):
        """
        Retorna información sobre el modelo
        """
        return {
            'model_name': Config.EMBEDDING_MODEL,
            'dimensions': self.model.get_sentence_embedding_dimension(),
            'max_seq_length': self.model.max_seq_length,
            'device': str(self.model.device)
        }


# Función auxiliar para testing
if __name__ == "__main__":
    # Test del modelo
    print("🧪 Probando modelo de embeddings...")

    embedder = EmbeddingModel()

    # Test 1: Encode simple
    text = "Me siento muy feliz y con energía"
    embedding = embedder.encode(text)
    print(f"✅ Embedding generado: shape {embedding.shape}")

    # Test 2: Batch encode
    texts = [
        "música alegre y energética",
        "canciones tristes y melancólicas",
        "ritmos para bailar"
    ]
    embeddings = embedder.encode_batch(texts)
    print(f"✅ Batch embeddings: shape {embeddings.shape}")

    # Test 3: Similitud
    text1 = "me siento feliz"
    text2 = "estoy muy contento"
    text3 = "estoy triste"

    sim_happy = embedder.get_similarity(text1, text2)
    sim_sad = embedder.get_similarity(text1, text3)

    print(f"✅ Similitud '{text1}' vs '{text2}': {sim_happy:.3f}")
    print(f"✅ Similitud '{text1}' vs '{text3}': {sim_sad:.3f}")

    # Info del modelo
    info = embedder.get_model_info()
    print(f"✅ Info del modelo: {info}")