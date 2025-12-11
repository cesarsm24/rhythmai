<img width="1500" height="180" alt="Logo RhythmAI" src="https://github.com/user-attachments/assets/d6254f2f-8ecb-4de0-977a-d5742bc3c67d" />

<p align="center">
  <br>
  <em>🎧 Tu Compañero Musical Inteligente Impulsado por IA y Análisis de Emociones</em><br>
  <em>Sistema avanzado de recomendación usando transformers, bases de datos vectoriales y búsqueda semántica.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30.0-ff4b4b?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-4.36-yellow?logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-0.4.22-7b2cbf?logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-1.7.4-00ADD8?logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Deezer%20API-Connected-00C7F2?logo=deezer&logoColor=white" />
  <img src="https://img.shields.io/badge/Security-AES--256-green?logo=lock&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>Español</strong>
</p>

---

## 📋 Tabla de Contenidos

- [Resumen](#-resumen)
- [Características](#-características)
- [Arquitectura Técnica](#-arquitectura-técnica)
- [Implementación de Base de Datos Vectorial](#-implementación-de-base-de-datos-vectorial)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Tests](#-tests)
- [Seguridad](#-seguridad)
- [Documentación de API](#-documentación-de-api)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Tecnologías](#-tecnologías)
- [Autores](#-autores)
- [Licencia](#-licencia)

---

## 🌟 Resumen

**RhythmAI** es un sistema de recomendación musical de última generación impulsado por IA que comprende las emociones y recomienda la banda sonora perfecta para el estado de ánimo. Construido con tecnologías de IA de vanguardia, combina inteligencia emocional, búsqueda semántica y bases de datos vectoriales para ofrecer experiencias musicales personalizadas.

### Cómo Funciona

```
Entrada Usuario → Análisis Emoción → Vectorización → Búsqueda Semántica → Recomendaciones → Aprendizaje Memoria
```

1. 🗣️ **El usuario describe su estado emocional** (lenguaje natural)
2. 🧠 **IA analiza emociones** usando transformer RoBERTa (28 categorías de emociones)
3. 🔢 **Vectorización de texto** con Sentence-BERT (embeddings de 384 dimensiones)
4. 🔍 **Búsqueda de similitud semántica** en base de datos vectorial ChromaDB/FAISS
5. 🎵 **Recomendaciones musicales** desde base de datos vectorial local
6. 💾 **Sistema de aprendizaje** recuerda preferencias para futuras sesiones

---

## ✨ Características

### 🎭 Análisis Avanzado de Emociones
- **28 Categorías de Emociones**: Alegría, tristeza, ira, miedo, entusiasmo, optimismo y más
- **Puntuación de Confianza**: Cada detección de emoción incluye porcentaje de confianza
- **Detección Multi-Emocional**: Reconoce estados emocionales complejos
- **Dimensiones de Energía y Valencia**: Cuantifica el estado de ánimo musical en dos ejes (escala 0-1)

### 🔍 Base de Datos Vectorial y Búsqueda Semántica
- **Soporte Dual de Vector Store**: Elección entre **ChromaDB** o **FAISS**
- **Almacenamiento de Alta Dimensión**: Vectores de embedding de 384 dimensiones
- **Búsqueda por Similitud de Coseno**: Encuentra canciones semánticamente similares
- **Vectorización por Lotes**: Procesamiento eficiente de listas de reproducción grandes
- **Filtrado por Metadatos**: Búsqueda por género, estado de ánimo o contexto
- **Indexación HNSW**: Búsqueda rápida de vecinos más cercanos aproximados
- **Rendimiento**: FAISS es 10-100x más rápido para conjuntos de datos grandes

### 🔐 Seguridad de Nivel Empresarial
- **Cifrado AES-256**: Cifrado de datos de grado militar
- **Derivación de Claves PBKDF2**: Generación segura de claves basadas en contraseñas (100,000 iteraciones)
- **Almacenamiento Cifrado**: Perfiles de usuario e historial de conversación seguros
- **Privacidad de Datos**: Ninguna información sensible almacenada en texto plano

### 🎵 Sistema de Recomendación Inteligente
- **Consciente del Contexto**: Comprende situaciones (ejercicio, estudio, fiesta, dormir)
- **Aprendizaje de Preferencias**: Mejora recomendaciones con el tiempo
- **Mapeo de Géneros**: Sugiere géneros óptimos basados en emociones
- **Integración Deezer**: Fuente de música vía API de Deezer
- **Mecanismos de Respaldo**: Manejo robusto de errores con estrategias alternativas

### 💬 Interfaz de Usuario Profesional
- **Diseño Moderno**: Fondos degradados, efectos glassmorphism
- **Diseño Responsive**: Funciona en escritorio y móvil
- **Elementos Animados**: Transiciones suaves y efectos hover
- **Previsualizaciones de Audio**: Escucha previsualizaciones de pistas de 30 segundos
- **Análisis Visuales**: Desgloses de emociones y estadísticas
- **Retroalimentación en Tiempo Real**: Indicadores de progreso y estados de carga

---

## 🏗️ Arquitectura Técnica

### Diagrama de Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                   Capa Frontend (Streamlit)                     │
│  ┌──────────────┬─────────────────┬──────────────────────────┐ │
│  │ Entrada      │ Visualizaciones │ Reproducción y           │ │
│  │ Usuario      │ y Análisis      │ Navegación               │ │
│  └──────────────┴─────────────────┴──────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│           Capa de Aplicación (MusicRecommender)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  • Orquestación de Peticiones  • Gestión de Contexto    │  │
│  │  • Ajuste de Preferencias      • Generación Respuestas  │  │
│  │  • Integración Memoria         • Manejo de Errores      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────┬──────────┬──────────────┬──────────────┬──────────┬──────┘
      │          │              │              │          │
┌─────▼───┐ ┌───▼───────┐ ┌────▼────────┐ ┌──▼──────┐ ┌▼──────────┐
│Analizador│ │  Modelo   │ │   Base de   │ │ Deezer  │ │  Módulo   │
│Emociones │ │ Embedding │ │  Datos Vec. │ │   API   │ │ Seguridad │
│          │ │           │ │ (ChromaDB)  │ │         │ │           │
│RoBERTa   │ │Sentence-  │ │ Índice HNSW │ │ Web API │ │  AES-256  │
│GoEmotions│ │BERT       │ │ Sim. Coseno │ │ Client  │ │  PBKDF2   │
└─────────┘ └───────────┘ └─────────────┘ └─────────┘ └───────────┘
```

---

## 💾 Implementación de Base de Datos Vectorial

### Resumen

RhythmAI soporta implementaciones **duales de base de datos vectorial**: **ChromaDB** (predeterminada) y **FAISS**, proporcionando flexibilidad entre facilidad de uso y rendimiento.

### Elegir un Vector Store

| Característica | ChromaDB | FAISS |
|----------------|----------|-------|
| **Velocidad** | Rápido (<50ms para 10K canciones) | Ultra-rápido (10-100x más rápido) |
| **Filtrado de Metadatos** | Soporte nativo | Implementación manual |
| **Configuración** | Configuración cero | Configuración mínima |
| **Escalabilidad** | Buena (10K-100K canciones) | Excelente (millones de vectores) |
| **Uso de Memoria** | Moderado | Bajo |
| **Mejor Para** | Uso general, desarrollo | Producción, conjuntos de datos grandes |

Cambia entre stores configurando `VECTOR_STORE=chroma` o `VECTOR_STORE=faiss` en `.env`.

---

## 🚀 Instalación

### Prerequisitos

- Python 3.9 o superior
- Gestor de paquetes pip
- 4GB RAM mínimo (8GB recomendado)
- 2GB espacio libre en disco

### Paso 1: Clonar Repositorio

```bash
git clone https://github.com/cesarsm24/rhythmai.git
cd rhythmai
```

### Paso 2: Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (macOS/Linux)
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Primera instalación** descargará modelos de IA (~1GB):
- `sentence-transformers/all-MiniLM-L6-v2` (80MB)
- `SamLowe/roberta-base-go_emotions` (500MB)

### Paso 4: Configurar Entorno

Crear archivo `.env` en la raíz del proyecto:

```env
# Selección de Vector Store
VECTOR_STORE=chroma  # Opciones: "chroma" o "faiss"
CHROMA_DB_PATH=./chroma_db
FAISS_DB_PATH=./faiss_db

# Modelos de IA
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMOTION_MODEL=SamLowe/roberta-base-go_emotions

# Configuración de Memoria
MEMORY_PATH=./memory
MAX_CONVERSATION_HISTORY=50
MEMORY_WINDOW=10

# Seguridad (Producción)
RHYTHM_MASTER_KEY=tu_clave_maestra_segura_cambiar_en_produccion

# Hardware (Opcional)
USE_GPU=false  # Establecer true para GPU habilitada con CUDA
```

### Paso 5: Poblar Base de Datos Vectorial

```bash
python scripts/populate_db.py
```

Este script:
- Obtiene listas de reproducción de Deezer (estados de ánimo configurados)
- Genera embeddings para canciones
- Almacena vectores en ChromaDB/FAISS
- Toma aproximadamente 5-10 minutos

---

## ⚙️ Configuración

### Variables de Entorno

| Variable | Descripción | Por Defecto |
|----------|-------------|-------------|
| `VECTOR_STORE` | Base de datos vectorial ("chroma" o "faiss") | `chroma` |
| `CHROMA_DB_PATH` | Ruta de almacenamiento ChromaDB | `./chroma_db` |
| `FAISS_DB_PATH` | Ruta de almacenamiento FAISS | `./faiss_db` |
| `EMBEDDING_MODEL` | Modelo de sentence transformer | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMOTION_MODEL` | Modelo de análisis de emociones | `SamLowe/roberta-base-go_emotions` |
| `MEMORY_PATH` | Ruta de memoria de usuario | `./memory` |
| `MAX_CONVERSATION_HISTORY` | Máx. conversaciones almacenadas | `50` |
| `MEMORY_WINDOW` | Tamaño de ventana de contexto | `10` |
| `USE_GPU` | Habilitar aceleración GPU | `false` |
| `RHYTHM_MASTER_KEY` | Clave maestra de cifrado | `default_key` |

---

## 💻 Uso

### Iniciar la Aplicación

```bash
streamlit run app.py
```

La aplicación se abre en: `http://localhost:8501`

### Usar el Sistema

#### Paso 1: Describir Estado de Ánimo

Ejemplos de prompts efectivos:

**Estados Emocionales:**
- "Me siento con energía y quiero bailar"
- "Estoy triste y necesito música tranquila"
- "Sintiéndome nostálgico del pasado"

**Basados en Actividad:**
- "Necesito música para concentrarme estudiando"
- "Lista de reproducción de entrenamiento de alta intensidad"
- "Música relajante para meditación"

**Específicos del Contexto:**
- "Conduciendo en un viaje largo por carretera"
- "Organizando una cena"
- "Preparándome para dormir"

#### Paso 2: Obtener Recomendaciones

Hacer clic en el botón **"🎵 Obtener Recomendaciones"**

El sistema:
1. Analiza estado emocional (2-3 segundos)
2. Busca en base de datos vectorial canciones similares
3. Devuelve recomendaciones personalizadas
4. Actualiza perfil de preferencias

---

## 🧪 Tests

### Ejecutar Todos los Tests

```bash
# Ejecutar todos los tests
pytest

# Ejecutar con cobertura
pytest --cov=rhythmai --cov-report=html

# Ejecutar tests específicos
pytest tests/unit/                    # Solo tests unitarios
pytest tests/integration/             # Solo tests de integración
```

### Ejecutar Tests por Categoría

```bash
# Tests unitarios
pytest -m unit

# Tests de integración
pytest -m integration

# Tests lentos
pytest -m slow
```

### Ver Reporte de Cobertura

```bash
# Generar reporte HTML
pytest --cov=rhythmai --cov-report=html

# Abrir en navegador (macOS)
open htmlcov/index.html

# Abrir en navegador (Linux)
xdg-open htmlcov/index.html
```

### Estructura de Tests

```
tests/
├── __init__.py
├── conftest.py                      # Fixtures compartidos
├── pytest.ini                       # Configuración pytest
│
├── unit/                            # Tests unitarios
│   ├── test_embeddings.py          # Tests de modelo de embeddings
│   ├── test_emotion_analyzer.py    # Tests de analizador de emociones
│   └── test_vector_stores.py       # Tests de almacenes vectoriales
│
└── integration/                     # Tests de integración
    ├── test_music_recommender.py   # Tests del sistema completo
    └── test_end_to_end.py          # Tests end-to-end
```

### Ejecutar Tests Antes de Commit

```bash
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Ejecutar linting
flake8 rhythmai scripts app.py
black --check rhythmai scripts app.py
isort --check-only rhythmai scripts app.py

# Ejecutar tests
pytest --cov=rhythmai

# Todo en uno
flake8 rhythmai && black --check rhythmai && pytest --cov=rhythmai
```

---

## 🔐 Seguridad

### Implementación de Cifrado

RhythmAI implementa cifrado de grado militar para datos sensibles:

#### Detalles del Algoritmo
- **Cifrado**: AES-256 (Advanced Encryption Standard)
- **Modo**: Fernet (cifrado simétrico)
- **Derivación de Claves**: PBKDF2-HMAC-SHA256
- **Iteraciones**: 100,000 (protección contra fuerza bruta)
- **Salt**: Estático por instalación (personalizar en producción)

### Ejemplos de Uso

#### Cifrar Datos String

```python
from rhythmai.utils.security import DataEncryption

encryptor = DataEncryption("tu_contraseña_maestra")

# Cifrar
datos_sensibles = "user_api_token_xyz123"
cifrado = encryptor.encrypt_string(datos_sensibles)

# Descifrar
descifrado = encryptor.decrypt_string(cifrado)
```

---

## 📚 Documentación de API

### Clase MusicRecommender

Orquestador principal para recomendaciones musicales.

#### Métodos

##### `recommend(user_input: str, n_results: int = 8) -> dict`

Generar recomendaciones musicales personalizadas.

**Parámetros:**
- `user_input` (str): Descripción del estado emocional del usuario
- `n_results` (int): Número de recomendaciones (predeterminado: 8)

**Retorna:**
```python
{
    'emotion_analysis': {
        'dominant_emotion': str,           # Emoción primaria
        'dominant_score': float,           # Confianza (0-1)
        'top_emotions': List[Dict],        # Top 5 emociones
        'dimensions': {
            'valence': float,              # Positividad (0-1)
            'energy': float                # Nivel de energía (0-1)
        },
        'suggested_genres': List[str],     # Géneros recomendados
        'context': List[str]               # Contextos detectados
    },
    'vector_results': List[Dict],          # Coincidencias BD vectorial
    'context_playlists': List[Dict],       # Listas contextuales
    'explanation': str,                    # Explicación en lenguaje natural
    'enriched_context': Dict               # Historial y preferencias
}
```

**Ejemplo:**
```python
from rhythmai.core.music_recommender import MusicRecommender

recommender = MusicRecommender(user_id="usuario123")

resultados = recommender.recommend(
    user_input="Me siento feliz y con energía",
    n_results=10
)

print(resultados['emotion_analysis']['dominant_emotion'])  # "joy"
print(len(resultados['vector_results']))                   # 10
```

---

## 📂 Estructura del Proyecto

```
rhythmai/
├── app.py                              # Aplicación web Streamlit
├── requirements.txt                    # Dependencias Python
├── requirements-dev.txt                # Dependencias de desarrollo
├── pytest.ini                          # Configuración pytest
├── .env                                # Variables de entorno (git-ignored)
├── .env.example                        # Plantilla de entorno
├── README.md                           # Este archivo (Inglés)
├── README_ES.md                        # Este archivo (Español)
│
├── rhythmai/                           # Paquete principal
│   ├── __init__.py
│   ├── config.py                       # Configuración centralizada
│   │
│   ├── core/                           # Módulos núcleo AI/ML
│   │   ├── __init__.py
│   │   ├── embeddings.py               # Embeddings Sentence-BERT
│   │   ├── emotion_analyzer.py         # Detección de emociones RoBERTa
│   │   ├── music_recommender.py        # Orquestador principal
│   │   └── deezer_client.py            # Wrapper API Deezer
│   │
│   ├── stores/                         # Implementaciones BD vectoriales
│   │   ├── __init__.py
│   │   ├── base_store.py               # Clase base abstracta
│   │   ├── factory.py                  # Patrón Factory (ChromaDB/FAISS)
│   │   ├── chroma_store.py             # Implementación ChromaDB
│   │   └── faiss_store.py              # Implementación FAISS
│   │
│   ├── memory/                         # Sistema de contexto y memoria
│   │   ├── __init__.py
│   │   ├── context_manager.py          # Orquestación de contexto
│   │   ├── conversation_memory.py      # Historial de conversación
│   │   └── user_profile.py             # Preferencias de usuario
│   │
│   └── utils/                          # Módulos de utilidad
│       ├── __init__.py
│       └── security.py                 # Cifrado/descifrado AES-256
│
├── scripts/                            # Scripts de utilidad
│   ├── populate_db.py                  # Script de población de BD
│   ├── clear_db.py                     # Limpiar base de datos vectorial
│   └── clear_memory.py                 # Limpiar memoria de usuario
│
├── tests/                              # Suite de tests
│   ├── __init__.py
│   ├── conftest.py                     # Fixtures compartidos
│   ├── unit/                           # Tests unitarios
│   │   ├── test_embeddings.py
│   │   ├── test_emotion_analyzer.py
│   │   └── test_vector_stores.py
│   └── integration/                    # Tests de integración
│       ├── test_music_recommender.py
│       └── test_end_to_end.py
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions CI/CD
│
├── docs/
│   └── architecture.md                 # Documentación de arquitectura
│
├── chroma_db/                          # Almacenamiento ChromaDB (git-ignored)
├── faiss_db/                           # Almacenamiento FAISS (git-ignored)
├── memory/                             # Memoria de usuario (git-ignored)
└── .cache/                             # Caché de modelos (git-ignored)
```

---

## 🛠️ Tecnologías

### IA y Machine Learning

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Transformers** | 4.36.2 | Análisis de emociones con RoBERTa |
| **Sentence-Transformers** | 2.3.1 | Embeddings de texto (384D) |
| **PyTorch** | 2.1.2 | Backend de deep learning |
| **NumPy** | 1.26.4 | Computación numérica |
| **scikit-learn** | 1.3.2 | Utilidades y métricas ML |

### Base de Datos Vectorial y Búsqueda

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **ChromaDB** | 0.4.18 | Almacenamiento de base de datos vectorial |
| **FAISS** | 1.7.4 | Búsqueda de similitud ultrarrápida |
| **HNSW** | 0.7.0 | Búsqueda de similitud rápida |

### APIs e Integración

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Requests** | 2.31.0 | Librería HTTP |
| **Python-dotenv** | 1.0.0 | Gestión de entorno |

### Seguridad

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Cryptography** | 41.0.7 | Cifrado AES-256 |
| **Pydantic** | 2.5.3 | Validación de datos |

### Web y UI

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Streamlit** | 1.30.0 | Framework de aplicación web |
| **Pandas** | 2.1.4 | Manipulación de datos |

### Herramientas de Desarrollo

| Herramienta | Propósito |
|-------------|-----------|
| **pytest** | Framework de testing |
| **black** | Formateo de código |
| **flake8** | Linting |
| **mypy** | Verificación de tipos |
| **isort** | Ordenamiento de imports |

---

## 👥 Autores

<div align="center">

| Autor | GitHub | Rol |
|:------|:------:|:----|
| **César Sánchez Montes** | [![GitHub](https://img.shields.io/badge/GitHub-cesarsm24-181717?style=flat&logo=github)](https://github.com/cesarsm24) | Desarrollador Principal, Arquitectura IA |
| **Miguel Ángel Campón Iglesias** | [![GitHub](https://img.shields.io/badge/GitHub-miguelit011-181717?style=flat&logo=github)](https://github.com/miguelit011) | Desarrollo Backend, Integración API |
| **Nicolás Benito Benito** | [![GitHub](https://img.shields.io/badge/GitHub-niconave17-181717?style=flat&logo=github)](https://github.com/niconave17) | Desarrollo Frontend, Diseño UI/UX |

</div>

---

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor, sigue estas pautas:

### Cómo Contribuir

1. **Hacer Fork** del repositorio
2. **Crear** rama de característica (`git checkout -b feature/caracteristica-increible`)
3. **Hacer Commit** de cambios (`git commit -m 'Añadir característica increíble'`)
4. **Hacer Push** a la rama (`git push origin feature/caracteristica-increible`)
5. **Abrir** un Pull Request

### Pautas de Desarrollo

- Seguir guía de estilo **PEP 8**
- Añadir **docstrings** a todas las funciones (estilo Google)
- Escribir **tests unitarios** para nuevas características
- Actualizar **documentación** según sea necesario
- Usar **type hints** donde sea aplicable
- Ejecutar **linting** antes de hacer commit

---

## 📄 Licencia

Este proyecto está licenciado bajo la **Licencia MIT**.

### Resumen Licencia MIT

✅ **Permisos:**
- Uso comercial
- Modificación
- Distribución
- Uso privado

⚠️ **Condiciones:**
- Aviso de licencia y copyright

❌ **Limitaciones:**
- Responsabilidad
- Garantía

Ver archivo [LICENSE](./LICENSE) para detalles completos.

---

## 🙏 Agradecimientos

- [**Deezer**](https://www.deezer.com/) - Por la completa API de música
- [**Hugging Face**](https://huggingface.co/) - Por alojar modelos transformer
- [**ChromaDB**](https://www.trychroma.com/) - Por la tecnología de base de datos vectorial
- [**Sentence-Transformers**](https://www.sbert.net/) - Por embeddings semánticos
- [**Streamlit**](https://streamlit.io/) - Por desarrollo rápido de aplicaciones web

---

## 🐛 Resolución de Problemas

### Problemas Comunes

#### Problema: Errores de "Module not found"
**Solución**: Asegurar que el entorno virtual esté activado y las dependencias instaladas:
```bash
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install -r requirements.txt
```

#### Problema: Fallos de tests
**Solución**:
```bash
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Ejecutar tests con verbose
pytest -v

# Verificar instalación de paquetes
pip list
```

#### Problema: ChromaDB se bloquea o corrompe
**Solución**:
```bash
rm -rf chroma_db/  # Eliminar base de datos
python scripts/populate_db.py  # Reconstruir
```

---

## 📞 Contacto y Soporte

**Repositorio del Proyecto**: [github.com/cesarsm24/rhythmai](https://github.com/cesarsm24/rhythmai)

**Reportar Problemas**: [GitHub Issues](https://github.com/cesarsm24/rhythmai/issues)

**Discusiones**: [GitHub Discussions](https://github.com/cesarsm24/rhythmai/discussions)

---

<div align="center">

**Hecho con ❤️ y 🎵 por el Equipo RhythmAI**

⭐ **¡Dale estrella a este repositorio si te resulta útil!**

![visitors](https://visitor-badge.laobi.icu/badge?page_id=cesarsm24.rhythmai)

</div>