<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=280&color=0:1e1b4b,50:6d28d9,100:a78bfa&text=MovieRec&fontColor=ffffff&fontSize=65&animation=fadeIn&fontAlignY=40"/>
</p>

<p align="center">
  <h1 align="center">🎬 MovieRec</h1>
  <p align="center">
    Interactive Cinema Recommendation System & Dashboard
  </p>
  <p align="center">
    TF-IDF Cosine Similarity • FastAPI Backend • Streamlit UI • Live TMDB Integration • Dynamic Local Fallback Mode
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/alokspacy/Movie-Recommend-Model?style=for-the-badge">
  <img src="https://img.shields.io/github/forks/alokspacy/Movie-Recommend-Model?style=for-the-badge">
  <img src="https://img.shields.io/github/license/alokspacy/Movie-Recommend-Model?style=for-the-badge">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi">
  <img src="https://img.shields.io/badge/Streamlit-Frontend-ff4b4b?style=flat-square&logo=streamlit">
  <img src="https://img.shields.io/badge/Pandas-Data--Preprocessing-150458?style=flat-square&logo=pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-TF--IDF--Model-f7931e?style=flat-square&logo=scikitlearn">
  <img src="https://img.shields.io/badge/TMDB-API--Integration-01b4e4?style=flat-square&logo=themoviedb">
</p>

---

# 📖 Overview

**MovieRec** is a cinematic movie recommendation dashboard engineered and developed by **Alok Singh**. The application uses a hybrid TF-IDF Content-Based Filtering engine alongside live TMDB API integrations to supply recommendations, trailers, genre suggestions, and rich details for over 45,000 films.

To guarantee zero friction, the application implements a robust **Local Fallback Mode**. If no TMDB API key is provided, the FastAPI backend will automatically parse the local `movies_metadata.csv` to perform searches, details extraction, and genre recommendations offline, mapping local poster paths to TMDB's public CDN so movie cards still render with images!

---

# ✨ Features

| Feature | Description |
|:---|:---|
| 🔎 **TF-IDF Content-Based Filtering** | Computes cosine similarity matrices on movie `overviews`, `genres`, and `taglines` to extract top matches. |
| 🛡️ **Zero-Config Local Fallback** | Runs fully offline without any API key by compiling index maps from `movies_metadata.csv` on startup. |
| 🎥 **Live TMDB Mode** | Automatically fetches top trending, popular, upcoming, and top-rated movies directly from TMDB when an API key is present. |
| 🔮 **Cinematic Dark Dashboard** | High-fidelity frontend built in Streamlit featuring outfit typography, radial background glows, and glassmorphic card layouts with custom glowing border transitions. |
| 📦 **Dual-Process Microservices** | Fully decoupled FastAPI backend and Streamlit frontend communicating through asynchronous REST endpoints. |

---

# 🛠️ Tech Stack

- **Backend**: Python 3.13, FastAPI, Uvicorn, HTTPX, python-dotenv
- **Frontend**: Streamlit, Custom CSS Injection (Glassmorphic Columns via `:has()`)
- **Machine Learning & Modeling**: Pandas, NumPy, Scikit-Learn (TF-IDF Vectorizer), SciPy (Sparse Matrices), Pickle Persistence
- **Datasets**: TMDB Movie Lens (45,000+ records)

---

# 🚀 Setup & Execution

### 1. Installation & Environment Setup
Clone the repository and set up a Python virtual environment:
```bash
# Set up virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory :
```ini
# The Movie Database (TMDB) API Key (Optional)
# If blank, the server runs in offline fallback mode using local metadata.
TMDB_API_KEY=

# Base URL of the backend service
API_BASE=http://127.0.0.1:8000
```

### 3. Run the Microservices

Launch the **FastAPI Backend Service**:
```bash
uvicorn main:app --reload
```
*Note: If no TMDB key is provided, the terminal will print: `WARNING: TMDB_API_KEY is not set. Running in LOCAL OFFLINE FALLBACK MODE using movies_metadata.csv.`*

Launch the **Streamlit Frontend App**:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to explore!
