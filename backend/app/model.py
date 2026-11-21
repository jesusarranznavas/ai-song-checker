from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .mert import MERTConfig, MERTModel

MERT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "mert", "config.json")
MERT_SAMPLE_RATE = 24000

@dataclass
class ModelBundle:
    encoder: nn.Module
    encoder_sr: int
    head: nn.Module
    labels: Tuple[str, str]

class MERTWrapper(nn.Module):
    """Wrapper sobre MERT que expone la misma interfaz que Wav2Vec2.

    La arquitectura se construye desde la configuración local (backend/app/mert),
    sin descargar nada. Los pesos se cargan después desde un checkpoint local.
    """
    def __init__(self):
        super().__init__()
        config = MERTConfig.from_json_file(MERT_CONFIG_PATH)
        self.mert = MERTModel(config)

    @property
    def hidden_size(self) -> int:
        return self.mert.config.hidden_size

    def load_pretrained(self, path: str) -> None:
        """Carga pesos del encoder desde:
          - un checkpoint del proyecto (dict con claves "head"/"encoder"), o
          - un state_dict de MERT original (pytorch_model.bin).
        """
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "encoder" in state:
            self.load_state_dict(state["encoder"])
        else:
            self.mert.load_state_dict(state, strict=False)

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

def build_model(
    labels: Tuple[str, str] = ("human", "pure_ai"),
    encoder_weights: Optional[str] = None,
) -> ModelBundle:
    encoder = MERTWrapper()
    if encoder_weights:
        encoder.load_pretrained(encoder_weights)
    head = MLPHead(encoder.hidden_size, num_classes=len(labels))
    return ModelBundle(encoder=encoder, encoder_sr=MERT_SAMPLE_RATE, head=head, labels=labels)

def encode_mean(encoder: nn.Module, wav: torch.Tensor) -> torch.Tensor:
    feats, _ = encoder.extract_features(wav)
    last = feats[-1]  # [B, T, C]
    return last.mean(dim=1)  # [B, C]
