
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .infer import AutocompleteEngine
except ImportError:  # pragma: no cover
    from infer import AutocompleteEngine


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

app = FastAPI(title="Autocomplete", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None


class PredictRequest(BaseModel):
    text: str
    top_k: int = 5
    temperature: float = 0.8


@app.on_event("startup")
def startup_event():
    global engine
    model_path = ARTIFACT_DIR / "model.pt"
    config_path = ARTIFACT_DIR / "config.json"
    vocab_path = ARTIFACT_DIR / "vocab.json"

    if model_path.exists() and config_path.exists() and vocab_path.exists():
        engine = AutocompleteEngine(ARTIFACT_DIR)
    else:
        engine = None


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_ready": engine is not None,
        "artifact_dir": str(ARTIFACT_DIR),
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run backend/train.py first.",
        )

    text = payload.text or ""
    if not text.strip():
        return {
            "prefix": text,
            "ghost_text": "",
            "next_word": "",
            "candidates": [],
        }

    result = engine.predict(text=text, top_k=max(1, min(payload.top_k, 10)), temperature=payload.temperature)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
