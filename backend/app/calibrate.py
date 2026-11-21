import json
import os
from typing import Optional

import torch

def load_temperature(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj.get("temperature", 1.0))

def apply_temperature(logits: torch.Tensor, temperature: Optional[float]) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        return logits
    return logits / temperature
