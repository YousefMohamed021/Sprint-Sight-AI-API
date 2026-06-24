"""
SprintSight — FastAPI Inference Server
Predicts sprint Productivity and Quality from structured features + sprint text.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from app.schemas import SprintInput, PredictionResponse, HealthResponse
from app.predictor import Predictor


# ── Lifespan: load models ONCE at startup, reuse for every request ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀  Loading models and tokenizer...")
    t0 = time.time()
    app.state.predictor = Predictor()          # loads all .pkl files + BERTOverflow
    print(f"✅  Ready in {time.time() - t0:.1f}s")
    yield
    print("👋  Shutting down")


# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SprintSight API",
    description=(
        "Predicts Agile sprint **Productivity** (completion ratio) and "
        "**Quality** (reopen ratio) using BERTOverflow embeddings + Random Forest. "
        "Based on Sprint2Vec (IEEE TSE 2025)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from any frontend (React, Streamlit, etc.) on localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "SprintSight API is running",
        "docs": "http://localhost:8000/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Check that all models are loaded and ready."""
    predictor: Predictor = app.state.predictor
    return HealthResponse(
        status="ok",
        models_loaded=predictor.is_ready(),
        embedding_model=predictor.embedding_model_id,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(sprint: SprintInput):
    """
    Submit sprint data and receive productivity + quality predictions.

    - **productivity**: predicted ratio of committed issues that will be completed (higher = better)
    - **quality**: predicted ratio of completed issues that will be reopened (lower = better)
    """
    predictor: Predictor = app.state.predictor

    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Models not yet loaded. Try again.")

    try:
        result = predictor.predict(sprint)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
