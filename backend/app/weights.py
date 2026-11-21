"""Descarga de los pesos entrenados desde GitHub Releases si no existen en local."""
from __future__ import annotations
import os
import sys
import urllib.request

WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))
MODEL_PATH = os.path.join(WEIGHTS_DIR, "model.pt")
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/jesusarranznavas/ai-song-checker/releases/download/v1.0/model.pt",
)


def ensure_weights(path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading model weights from {url} ...", flush=True)
    tmp = path + ".part"

    def _progress(blocks, block_size, total):
        if total > 0:
            pct = min(100.0, blocks * block_size * 100.0 / total)
            sys.stdout.write(f"\r  {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    os.replace(tmp, path)
    print(f"\nSaved -> {path}", flush=True)
    return path


if __name__ == "__main__":
    ensure_weights()
