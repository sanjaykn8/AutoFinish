"""
app.py
──────
FastAPI backend for the hybrid autocomplete system.

Endpoints:
  GET  /health           → model status
  POST /predict          → autocomplete predictions
  GET  /training-history → loss curves for the dashboard
  GET  /ngram-summary    → n-gram training summary
  GET  /model-info       → model architecture details

Run:
  uvicorn backend.app:app --reload --port 8000
  # or:
  python backend/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .infer import AutocompleteEngine
except ImportError:
    from infer import AutocompleteEngine

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hybrid Autocomplete API",
    description="Character-level LSTM + Word-level N-gram autocomplete engine.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: AutocompleteEngine | None = None


# ─── Models ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., description="Current editor text")
    top_k: int = Field(5, ge=1, le=10, description="Number of candidates to return")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event() -> None:
    global engine
    required = [ARTIFACT_DIR / "model.pt", ARTIFACT_DIR / "config.json", ARTIFACT_DIR / "vocab.json"]
    optional = [ARTIFACT_DIR / "ngram_model.json"]

    has_required = all(p.exists() for p in required)
    has_ngram = all(p.exists() for p in optional)

    if not has_required and not has_ngram:
        print("[api] No model artifacts found. Serving in degraded mode.")
        engine = None
        return

    try:
        engine = AutocompleteEngine(ARTIFACT_DIR)
        status = engine.model_status()
        print(f"[api] Engine loaded — char={status['char_model']}, ngram={status['ngram_model']}")
    except Exception as e:
        print(f"[api] Failed to load engine: {e}")
        engine = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    status = engine.model_status() if engine else {"char_model": False, "ngram_model": False}
    return {
        "ok": True,
        "model_ready": engine is not None,
        "char_model": status.get("char_model", False),
        "ngram_model": status.get("ngram_model", False),
        "artifact_dir": str(ARTIFACT_DIR),
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Run train.py and train_ngram.py first.",
        )
    result = engine.predict(
        text=payload.text,
        top_k=payload.top_k,
        temperature=payload.temperature,
    )
    return result


@app.get("/training-history")
def training_history():
    path = ARTIFACT_DIR / "training_history.json"
    if not path.exists():
        return {"history": [], "available": False}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {"history": data, "available": True}


@app.get("/ngram-summary")
def ngram_summary():
    path = ARTIFACT_DIR / "ngram_summary.json"
    if not path.exists():
        return {"available": False}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {"available": True, **data}


@app.get("/model-info")
def model_info():
    config_path = ARTIFACT_DIR / "config.json"
    if not config_path.exists():
        return {"available": False}
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    ngram_path = ARTIFACT_DIR / "ngram_summary.json"
    ngram_info = {}
    if ngram_path.exists():
        with ngram_path.open("r", encoding="utf-8") as f:
            ngram_info = json.load(f)

    return {
        "available": True,
        "char_model": {
            "architecture": "CharLSTM",
            "vocab_size": config.get("vocab_size"),
            "embed_size": config.get("embed_size"),
            "hidden_size": config.get("hidden_size"),
            "num_layers": config.get("num_layers"),
            "dropout": config.get("dropout"),
            "seq_len": config.get("seq_len"),
        },
        "ngram_model": {
            "architecture": f"{ngram_info.get('n', 3)}-gram with Kneser-Ney smoothing",
            **ngram_info,
        },
    }


# ─── Dev entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
