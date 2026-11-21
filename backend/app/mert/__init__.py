"""
Vendored copy of the MERT-v1-95M architecture (m-a-p, "MERT: Acoustic Music
Understanding Model with Large-Scale Self-supervised Training", Li et al., 2023).

Source: m-a-p/MERT-v1-95M, licensed CC BY-NC 4.0.
Only the model definition and config are included here; no network access is
needed to build the model. The weights are loaded from backend/weights/model.pt.
"""
from .configuration_MERT import MERTConfig
from .modeling_MERT import MERTModel

__all__ = ["MERTConfig", "MERTModel"]
