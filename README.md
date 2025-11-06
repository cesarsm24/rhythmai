<img width="1500" height="180" alt="Logo RhythmAI" src="https://github.com/user-attachments/assets/d6254f2f-8ecb-4de0-977a-d5742bc3c67d" />

<p align="center">
  <br>
  <em>🎧 Asistente musical inteligente basado en emociones</em><br>
  <em>Impulsado por IA, embeddings semánticos y la API de Spotify.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30.0-ff4b4b?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1.0-3A86FF?logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-0.4.22-7b2cbf?logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/Spotify%20API-Connected-1ED760?logo=spotify&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## 🧠 Funcionalidad principal

**RhythmAI** es un DJ virtual impulsado por inteligencia artificial que recomienda canciones según tu **estado emocional** o tu **contexto**.  

Usa **embeddings semánticos**, **bases de datos vectoriales (Chroma)** y la **API de Spotify** para ofrecerte música que encaje contigo, ya sea para **estudiar, relajarte o motivarte**.  

---

## 🎵 Flujo de funcionamiento

1. 🗣️ El usuario escribe cómo se siente o qué tipo de música desea.  
2. 🤖 RhythmAI convierte esa descripción en un **vector semántico** con *Sentence Transformers* o *OpenAI embeddings*.  
3. 💾 Se busca en una **base vectorial** de canciones con *ChromaDB*.  
4. 🎧 El sistema muestra los temas más similares y genera enlaces directos a **Spotify**.

---

## 🧩 Tecnologías Principales

<div align="center">

| Área | Herramienta |
|:----:|:------------|
| 🎵 Música | [Spotify Web API](https://developer.spotify.com/documentation/web-api) + Spotipy |
| 💬 IA Semántica | HuggingFace / Sentence Transformers |
| 🧠 Framework de IA | LangChain |
| 💾 Base Vectorial | ChromaDB |
| 🌐 Interfaz | Streamlit |
| 🧮 Utilidades | NumPy, Pandas |

</div>

---

## ⚙️ Instalación

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/cesarsm/rhythmai.git
cd RhythmAI
```

### 2️⃣ Crear entorno virtual e instalar dependencias
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Configurar variables de entorno
Crea un archivo .env en la raíz del proyecto con tus credenciales de Spotify:
```bash
SPOTIPY_CLIENT_ID=tu_client_id
SPOTIPY_CLIENT_SECRET=tu_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
```
---

## 🚀 Ejecución

Ejecuta la aplicación con:

```bash
streamlit run app.py
```
Esto abrirá la interfaz web en tu navegador (por defecto en http://localhost:8501).

---

## 🧾 Dependencias (requirements.txt)
```bash
# Core
spotipy==2.23.0
python-dotenv==1.0.0
streamlit==1.30.0

# AI/ML
sentence-transformers==2.3.1
transformers==4.36.0
torch==2.1.2

# LangChain
langchain==1.0.3
langchain-core==1.0.3
langchain-community==1.0.0a1

# Base de datos vectorial
chromadb==0.4.22

# Utilidades
pandas==2.1.4
numpy==1.26.3
```

---

## 🎨 Paleta de Colores y Estilo Visual

<div align="center">

| Color | Código | Uso |
|:-----:|:------:|:----|
| 🟢 **Verde Neón** | `#1ED760` | Energía y conexión con Spotify |
| ⚫ **Negro Profundo** | `#121212` | Fondo principal |
| 🟣 **Violeta Neón** | `#9B5DE5` | Creatividad y emoción |
| ⚪ **Gris Suave** | `#CCCCCC` | Texto secundario |

**Estilo visual:** Futurista, minimalista, *"neon chill"*

</div>

---

## 🧑‍💻 Autores

<div align="center">

| Autor | GitHub |
|:------|:------:|
| **César Sánchez Montes** | [![GitHub](https://img.shields.io/badge/GitHub-cesarsm24-181717?style=flat&logo=github)](https://github.com/cesarsm24) |
| **Miguel Ángel Campón Iglesias** | [![GitHub](https://img.shields.io/badge/GitHub-miguelit011-181717?style=flat&logo=github)](https://github.com/miguelit011) |
| **Nicolás Benito Benito** | [![GitHub](https://img.shields.io/badge/GitHub-nicolasbenito-181717?style=flat&logo=github)](https://github.com/nicolasbenito) |

</div>

---

## 📜 Licencia

Este proyecto se distribuye bajo la [licencia MIT](./LICENSE).
Eres libre de usarlo, modificarlo y compartirlo con atribución.



