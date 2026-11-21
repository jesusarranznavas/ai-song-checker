from __future__ import annotations
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import numpy as np

from .config import settings
from .audio import load_audio_bytes, window_audio
from .infer import InferenceEngine, entropy

app = FastAPI(title="AI Song Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InferenceEngine(labels=settings.binary_labels)


def _analyze_result(data: bytes, filename: str) -> dict:
    audio = load_audio_bytes(data, target_sr=settings.sample_rate)
    windows = window_audio(audio, settings.sample_rate, settings.window_sec, settings.stride_sec, settings.max_windows)
    probs3 = engine.predict_windows(windows, settings.sample_rate)
    p = np.array([probs3["human"], probs3["ai_modified"], probs3["pure_ai"]], dtype=np.float32)
    max_prob = float(p.max())
    ent = entropy(p)
    inconclusive = max_prob < settings.inconclusive_threshold
    return {
        "filename": filename,
        "duration_sec": float(len(audio) / settings.sample_rate),
        "window_sec": settings.window_sec,
        "stride_sec": settings.stride_sec,
        "num_windows": len(windows),
        "human": probs3["human"],
        "ai_modified": probs3["ai_modified"],
        "pure_ai": probs3["pure_ai"],
        "max_prob": max_prob,
        "entropy": ent,
        "inconclusive": inconclusive,
        "reason": "low_confidence" if inconclusive else "ok",
    }


# ── /health and /analyze (local dev, direct) ──
@app.get("/health")
def health():
    return {"status": "ok", "device": engine.device}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo inválido")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Archivo vacío")
    return _analyze_result(data, file.filename)


# ── /api/* (production: frontend calls /api/analyze) ──
@app.get("/api/health")
def api_health():
    return {"status": "ok", "device": engine.device}


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo inválido")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Archivo vacío")
    return _analyze_result(data, file.filename)


# ── Static frontend (production Docker) ──
_STATIC = os.path.join(os.path.dirname(__file__), "..", "..", "static")

if os.path.isdir(_STATIC):
    app.mount("/assets", StaticFiles(directory=os.path.join(_STATIC, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        candidate = os.path.join(_STATIC, full_path)
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_STATIC, "index.html"))
