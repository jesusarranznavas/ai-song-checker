from __future__ import annotations
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .model import build_model, encode_mean
from .calibrate import load_temperature, apply_temperature
from .weights import ensure_weights

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

class InferenceEngine:
    def __init__(self, device: Optional[str] = None, labels: Tuple[str, str] = ("human", "pure_ai")):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_bundle = build_model(labels=labels)
        self.model_bundle.encoder.to(self.device).eval()
        self.model_bundle.head.to(self.device).eval()

        self.temperature_path = os.path.join(WEIGHTS_DIR, "temperature.json")
        self.temperature = load_temperature(self.temperature_path)

        self.weights_path = ensure_weights(os.path.join(WEIGHTS_DIR, "model.pt"))
        self.load(self.weights_path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model_bundle.head.load_state_dict(ckpt["head"])
        if "encoder" in ckpt:
            self.model_bundle.encoder.load_state_dict(ckpt["encoder"])
        self.temperature = load_temperature(self.temperature_path)

    def _prepare_wav(self, wav_np: np.ndarray, sr: int) -> torch.Tensor:
        wav = torch.from_numpy(wav_np).float().unsqueeze(0)  # [1, T]
        if sr != self.model_bundle.encoder_sr:
            wav = torchaudio.functional.resample(wav, sr, self.model_bundle.encoder_sr)
        wav = torch.clamp(wav, -1.0, 1.0)
        return wav.to(self.device)

    @staticmethod
    def _mix_to_three_classes(p_h: float, p_ai: float) -> Dict[str, float]:
        """
        Convierte (human vs pure_ai) a 3 clases, forzando:
          - ai_modified = "intermedio" (mezcla) entre ambos
          - suma total = 1
        """
        p_h = float(np.clip(p_h, 0.0, 1.0))
        p_ai = float(np.clip(p_ai, 0.0, 1.0))
        # idealmente p_h + p_ai = 1 (softmax), pero por seguridad:
        s = p_h + p_ai
        if s <= 0:
            p_h, p_ai = 0.5, 0.5
        else:
            p_h, p_ai = p_h / s, p_ai / s

        # Pequeña corrección de desbalance (43 human vs 50 pure_ai en training)
        p_h = min(1.0, p_h + 0.04)
        total = p_h + p_ai
        p_h, p_ai = p_h / total, p_ai / total

        # ai_modified escala linealmente con la incertidumbre, máximo 0.5 en 50/50.
        # Esto hace que casos ambiguos queden por debajo del inconclusive_threshold
        # y se marquen como "inconclusive" en lugar de dominar con ai_modified.
        p_mod = (1.0 - abs(p_h - p_ai)) * 0.5

        # repartimos para que sume 1
        human_final = p_h * (1.0 - p_mod)
        pure_final = p_ai * (1.0 - p_mod)

        # clamp + renormalize final por seguridad
        vec = np.array([human_final, p_mod, pure_final], dtype=np.float32)
        vec = np.clip(vec, 0.0, 1.0)
        vec = vec / max(1e-9, float(vec.sum()))
        return {"human": float(vec[0]), "ai_modified": float(vec[1]), "pure_ai": float(vec[2])}

    @torch.inference_mode()
    def predict_windows(self, windows: List[np.ndarray], sr: int) -> Dict[str, float]:
        probs_bin_all = []
        for w in windows:
            wav = self._prepare_wav(w, sr)
            emb = encode_mean(self.model_bundle.encoder, wav)
            logits = self.model_bundle.head(emb)
            logits = apply_temperature(logits, self.temperature)
            probs_bin = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
            probs_bin_all.append(probs_bin)

        probs_mean = np.mean(np.stack(probs_bin_all, axis=0), axis=0)  # [2]
        # labels: ("human","pure_ai")
        p_h = float(probs_mean[0])
        p_ai = float(probs_mean[1])

        return self._mix_to_three_classes(p_h, p_ai)

def entropy(p: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))
