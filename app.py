"""
app.py
------
FastAPI web app for the Spotify Semantic Search Engine.

Run with:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 in a browser.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from search_engine import SemanticSearchEngine

# Model + Milvus connection loaded once at startup, shared across all requests.
engine: SemanticSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SemanticSearchEngine()
    stats = engine.client.get_collection_stats(engine.collection)
    print(f"Search engine ready: {stats.get('row_count', '?')} lyric chunks indexed.")
    yield
    engine.close()


app = FastAPI(title="Spotify Semantic Search Engine", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class SearchResult(BaseModel):
    artist: str
    song: str
    link: str
    score: float
    excerpt: str


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResult]


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={}
)

@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Natural-language vibe/mood query"),
    top_k: int = Query(10, ge=1, le=50),
):
    results = engine.search(q, top_k=top_k)
    return SearchResponse(query=q, count=len(results), results=results)


@app.get("/api/health")
def health():
    if not engine:
        return {"status": "starting"}
    stats = engine.client.get_collection_stats(engine.collection)
    return {"status": "ok", "chunks_indexed": stats.get("row_count", "?")}
