# AI Song Checker

Detects whether a song is **human-made or AI-generated** from its audio alone.

Final Degree Project (TFG) — Universidad Camilo José Cela, 2025/2026.

**Live demo:** [huggingface.co/spaces/JesusArranz/ai-song-checker](https://huggingface.co/spaces/JesusArranz/ai-song-checker)

![Home](docs/screenshots/home.png)

## How it works

1. The uploaded track is decoded with ffmpeg, converted to mono and resampled to 24 kHz.
2. Audio is split into fixed-length chunks and each one is encoded with **MERT-v1-95M**
   (Music Encoder Representations from Transformers).
3. Mean-pooled embeddings go through a small **MLP classification head** (`human` vs `pure_ai`).
4. Chunk predictions are aggregated and calibrated with **temperature scaling** to produce
   a trustworthy confidence score.

![Result](docs/screenshots/01-result.png)

## Tech stack

| Layer     | Technologies                                        |
|-----------|-----------------------------------------------------|
| Model     | PyTorch, Hugging Face Transformers (MERT), torchaudio, librosa |
| Backend   | FastAPI, Uvicorn                                    |
| Frontend  | React 18, Vite, React Router, lucide-react          |
| Deploy    | Docker (multi-stage build), Hugging Face Spaces     |

## Project structure

```
backend/
  app/
    main.py      # FastAPI app: /api/analyze, /api/health, serves the SPA
    infer.py     # Inference pipeline (chunking, aggregation, calibration)
    model.py     # MERT encoder + MLP head
    audio.py     # Audio decoding and preprocessing
    calibrate.py # Temperature-scaling utilities
  weights/       # temperature.json (model.pt is not versioned, see below)
training/
  dataset.py, augment.py, train.py, eval.py, baseline.py,
  temperature_scaling.py, export_onnx.py
frontend/react-src/   # React + Vite web app
Dockerfile            # Builds frontend and backend into a single image
```

## Running locally

> The trained weights (`backend/weights/model.pt`, ~380 MB) are not included in this
> repository because of GitHub's file size limits. They can be obtained from the
> [Hugging Face Space](https://huggingface.co/spaces/JesusArranz/ai-song-checker/tree/main/backend/weights),
> or you can train your own model with the scripts in `training/`.

### With Docker

```bash
docker build -t ai-song-checker .
docker run -p 7860:7860 ai-song-checker
# open http://localhost:7860
```

### Manually

```bash
# Backend (requires ffmpeg)
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --port 8000

# Frontend
cd frontend/react-src
npm install
npm run dev
```

## Training

Expected dataset layout:

```
data/
  train/{human,pure_ai}/*.wav|mp3
  val/{human,pure_ai}/...
  test/{human,pure_ai}/...
```

```bash
pip install -r training/requirements.txt
python -m training.train --data_dir ./data --epochs 10
python -m training.eval --data_dir ./data
python -m training.temperature_scaling --data_dir ./data
```

## Screenshots

| Analyzer | About |
|----------|-------|
| ![Analyzer](docs/screenshots/analyzer.png) | ![About](docs/screenshots/about.png) |

## Author

**Jesús Arranz** — [GitHub](https://github.com/jesusarranznavas)
