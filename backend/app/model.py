from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

MERT_MODEL_ID = "m-a-p/MERT-v1-95M"
MERT_SAMPLE_RATE = 24000

@dataclass
class ModelBundle:
    encoder: nn.Module
    encoder_sr: int
    head: nn.Module
    labels: Tuple[str, str]

class MERTWrapper(nn.Module):
    """Wrapper sobre MERT que expone la misma interfaz que Wav2Vec2."""
    def __init__(self):
        super().__init__()
        self.mert = AutoModel.from_pretrained(MERT_MODEL_ID, trust_remote_code=True)

    def extract_features(self, wav: torch.Tensor):
        outputs = self.mert(input_values=wav)
        # Devuelve tupla de un elemento para compatibilidad con encode_mean
        return (outputs.last_hidden_state,), None

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def build_model(labels: Tuple[str, str] = ("human", "pure_ai")) -> ModelBundle:
    encoder = MERTWrapper()
    encoder_sr = MERT_SAMPLE_RATE

    dummy = torch.zeros(1, encoder_sr)  # 1s a 24kHz
    with torch.inference_mode():
        feats, _ = encoder.extract_features(dummy)
        last = feats[-1]  # [B, T, C]
        in_dim = last.shape[-1]

    head = MLPHead(in_dim, num_classes=len(labels))
    return ModelBundle(encoder=encoder, encoder_sr=encoder_sr, head=head, labels=labels)

def encode_mean(encoder: nn.Module, wav: torch.Tensor) -> torch.Tensor:
    feats, _ = encoder.extract_features(wav)
    last = feats[-1]  # [B, T, C]
    return last.mean(dim=1)  # [B, C]
