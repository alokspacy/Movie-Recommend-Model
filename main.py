import os
import pickle
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# ENV & CONFIG
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

# Determine if we run in offline mode
IS_OFFLINE = not bool(TMDB_API_KEY)

if IS_OFFLINE:
    print("\n" + "=" * 60)
    print("WARNING: TMDB_API_KEY is not set.")
    print("Running in LOCAL OFFLINE FALLBACK MODE using movies_metadata.csv.")
    print("=" * 60 + "\n")
else:
    print("\n" + "=" * 60)
    print("TMDB_API_KEY detected. Running in live TMDB mode.")
    print("=" * 60 + "\n")

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PICKLE GLOBALS & METADATA CACHE
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

# Offline metadata lookup dictionaries
ID_TO_METADATA: Dict[int, Dict[str, Any]] = {}
TITLE_TO_METADATA: Dict[str, Dict[str, Any]] = {}

# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None

class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []

class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()

def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"

async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe TMDB GET:
    - Network errors -> 502
    - TMDB API errors -> 502 with detail
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()

async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out

async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )

async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )

async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None

# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )

def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )

def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out

async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    if IS_OFFLINE:
        norm_t = _norm_title(title)
        if norm_t in TITLE_TO_METADATA:
            rm = TITLE_TO_METADATA[norm_t]
            return TMDBMovieCard(
                tmdb_id=rm["tmdb_id"],
                title=rm["title"],
                poster_url=rm["poster_url"],
                release_date=rm["release_date"],
                vote_average=rm["vote_average"],
            )
        return None

    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None

# =========================
# STARTUP: LOAD PICKLES & METADATA
# =========================
@app.on_event("startup")
def startup_event():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    global ID_TO_METADATA, TITLE_TO_METADATA

    # 1. Load Pickles
    print("Loading TF-IDF pickles...")
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    # 2. Load metadata CSV if running offline
    if IS_OFFLINE:
        csv_path = os.path.join(BASE_DIR, "movies_metadata.csv")
        print(f"Loading {csv_path} for offline fallback mode...")
        try:
            import ast
            cols = ['id', 'title', 'poster_path', 'release_date', 'genres', 'overview', 'vote_average', 'popularity']
            meta = pd.read_csv(csv_path, usecols=cols, low_memory=False)
            
            # Clean dataframe
            meta = meta.dropna(subset=['id', 'title'])
            meta['id_numeric'] = pd.to_numeric(meta['id'], errors='coerce')
            meta = meta.dropna(subset=['id_numeric'])
            meta['id'] = meta['id_numeric'].astype(int)
            
            ID_TO_METADATA = {}
            TITLE_TO_METADATA = {}

            for _, row in meta.iterrows():
                tmdb_id = int(row['id'])
                title = str(row['title']).strip()
                p_path = row['poster_path']
                
                poster_url = f"{TMDB_IMG_500}{p_path}" if pd.notna(p_path) and str(p_path).startswith('/') else None
                # Fallback backdrop to poster_url
                backdrop_url = poster_url
                
                # Parse genres
                genres_list = []
                raw_genres = row['genres']
                if pd.notna(raw_genres) and str(raw_genres).strip():
                    try:
                        parsed = ast.literal_eval(raw_genres)
                        if isinstance(parsed, list):
                            genres_list = parsed
                    except Exception:
                        pass
                
                movie_dict = {
                    "tmdb_id": tmdb_id,
                    "title": title,
                    "poster_url": poster_url,
                    "backdrop_url": backdrop_url,
                    "release_date": str(row['release_date'])[:10] if pd.notna(row['release_date']) else None,
                    "overview": str(row['overview']) if pd.notna(row['overview']) else "",
                    "genres": genres_list,
                    "vote_average": float(row['vote_average']) if pd.notna(row['vote_average']) else 0.0,
                    "popularity": float(row['popularity']) if pd.notna(row['popularity']) else 0.0
                }
                
                ID_TO_METADATA[tmdb_id] = movie_dict
                TITLE_TO_METADATA[_norm_title(title)] = movie_dict

            print(f"Offline fallback initialized with {len(ID_TO_METADATA)} movies.")
        except Exception as e:
            print(f"ERROR: Failed to load movies_metadata.csv: {e}")
            print("Offline fallback will be limited.")

# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "mode": "offline" if IS_OFFLINE else "online"}

# ---------- HOME FEED ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed:
    - trending, popular, top_rated, upcoming, now_playing
    """
    if IS_OFFLINE:
        # Generate feed from local cached metadata
        movies = list(ID_TO_METADATA.values())
        if category == "top_rated":
            sorted_movies = sorted(movies, key=lambda x: x.get("vote_average", 0.0), reverse=True)
        else:
            # Sort by popularity for all other feeds in offline mode
            sorted_movies = sorted(movies, key=lambda x: x.get("popularity", 0.0), reverse=True)
        
        cards = []
        for m in sorted_movies[:limit]:
            cards.append(TMDBMovieCard(
                tmdb_id=m["tmdb_id"],
                title=m["title"],
                poster_url=m["poster_url"],
                release_date=m["release_date"],
                vote_average=m["vote_average"]
            ))
        return cards

    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")

# ---------- TMDB KEYWORD SEARCH ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    if IS_OFFLINE:
        q = _norm_title(query)
        matches = []
        for key, m in TITLE_TO_METADATA.items():
            if q in key:
                matches.append(m)
        
        matches = sorted(matches, key=lambda x: x.get("popularity", 0.0), reverse=True)
        
        results = []
        for m in matches[(page-1)*24 : page*24]:
            p_url = m["poster_url"]
            p_path = p_url.replace(TMDB_IMG_500, "") if p_url else None
            results.append({
                "id": m["tmdb_id"],
                "title": m["title"],
                "poster_path": p_path,
                "release_date": m["release_date"],
                "vote_average": m["vote_average"]
            })
        return {
            "results": results,
            "page": page,
            "total_results": len(matches),
            "total_pages": (len(matches) + 23) // 24
        }

    return await tmdb_search_movies(query=query, page=page)

# ---------- MOVIE DETAILS ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    if IS_OFFLINE:
        if tmdb_id in ID_TO_METADATA:
            m = ID_TO_METADATA[tmdb_id]
            return TMDBMovieDetails(
                tmdb_id=m["tmdb_id"],
                title=m["title"],
                overview=m["overview"],
                release_date=m["release_date"],
                poster_url=m["poster_url"],
                backdrop_url=m["backdrop_url"],
                genres=m["genres"]
            )
        raise HTTPException(status_code=404, detail="Movie not found in local dataset")

    return await tmdb_movie_details(tmdb_id)

# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    if IS_OFFLINE:
        if tmdb_id not in ID_TO_METADATA:
            return []
        m = ID_TO_METADATA[tmdb_id]
        if not m["genres"]:
            return []
        
        genre_name = m["genres"][0]["name"]
        matches = []
        for movie in ID_TO_METADATA.values():
            if movie["tmdb_id"] == tmdb_id:
                continue
            for g in movie["genres"]:
                if g["name"] == genre_name:
                    matches.append(movie)
                    break
        
        matches = sorted(matches, key=lambda x: x.get("popularity", 0.0), reverse=True)
        cards = []
        for movie in matches[:limit]:
            cards.append(TMDBMovieCard(
                tmdb_id=movie["tmdb_id"],
                title=movie["title"],
                poster_url=movie["poster_url"],
                release_date=movie["release_date"],
                vote_average=movie["vote_average"]
            ))
        return cards

    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]

# ---------- TF-IDF ONLY ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]

# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    if IS_OFFLINE:
        q = _norm_title(query)
        best_match = None
        if q in TITLE_TO_METADATA:
            best_match = TITLE_TO_METADATA[q]
        else:
            candidates = [m for key, m in TITLE_TO_METADATA.items() if q in key]
            if candidates:
                candidates = sorted(candidates, key=lambda x: x.get("popularity", 0.0), reverse=True)
                best_match = candidates[0]

        if not best_match:
            raise HTTPException(
                status_code=404, detail=f"No movie found locally for query: {query}"
            )

        tmdb_id = best_match["tmdb_id"]
        title = best_match["title"]

        details = TMDBMovieDetails(
            tmdb_id=tmdb_id,
            title=title,
            overview=best_match["overview"],
            release_date=best_match["release_date"],
            poster_url=best_match["poster_url"],
            backdrop_url=best_match["backdrop_url"],
            genres=best_match["genres"]
        )

        # 1) TF-IDF recommendations
        tfidf_items: List[TFIDFRecItem] = []
        recs: List[Tuple[str, float]] = []
        try:
            recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
        except Exception:
            try:
                recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
            except Exception:
                recs = []

        for r_title, score in recs:
            card = await attach_tmdb_card_by_title(r_title)
            tfidf_items.append(TFIDFRecItem(title=r_title, score=score, tmdb=card))

        # 2) Genre recommendations
        genre_recs: List[TMDBMovieCard] = []
        if details.genres:
            genre_name = details.genres[0]["name"]
            matches = []
            for movie in ID_TO_METADATA.values():
                if movie["tmdb_id"] == tmdb_id:
                    continue
                for g in movie["genres"]:
                    if g["name"] == genre_name:
                        matches.append(movie)
                        break
            matches = sorted(matches, key=lambda x: x.get("popularity", 0.0), reverse=True)
            for movie in matches[:genre_limit]:
                genre_recs.append(TMDBMovieCard(
                    tmdb_id=movie["tmdb_id"],
                    title=movie["title"],
                    poster_url=movie["poster_url"],
                    release_date=movie["release_date"],
                    vote_average=movie["vote_average"]
                ))

        return SearchBundleResponse(
            query=query,
            movie_details=details,
            tfidf_recommendations=tfidf_items,
            genre_recommendations=genre_recs
        )

    # Live Online Mode
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations
    tfidf_items = []
    recs = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
