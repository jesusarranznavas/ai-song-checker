# ── Stage 1: build React frontend ──────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /app
COPY frontend/react-src/package*.json ./
RUN npm install
COPY frontend/react-src/ ./
RUN npm run build

# ── Stage 2: Python backend ─────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Backend code (MERT architecture is vendored in backend/app/mert)
COPY backend/ ./backend/
COPY scripts/ ./scripts/

# Model weights: uses backend/weights/model.pt if present, otherwise downloads it
# from GitHub Releases
RUN python scripts/download_weights.py

# Built frontend
COPY --from=frontend-builder /app/dist ./static/

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
