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

# Pre-download MERT architecture (baked into image — no internet needed at runtime)
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('m-a-p/MERT-v1-95M', trust_remote_code=True)"

# Copy backend code and model weights
COPY backend/ ./backend/

# Copy built frontend
COPY --from=frontend-builder /app/dist ./static/

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
