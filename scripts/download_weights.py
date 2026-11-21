"""Descarga backend/weights/model.pt desde GitHub Releases.

Uso (desde la raíz del repo):  python scripts/download_weights.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.weights import ensure_weights  # noqa: E402

if __name__ == "__main__":
    ensure_weights()
