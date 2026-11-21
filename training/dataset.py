from __future__ import annotations
import os
import glob
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa

LABELS = ("human", "pure_ai")

AUDIO_EXTS = ("*.wav","*.mp3","*.flac","*.ogg","*.m4a","*.aac")

def list_audio_files(root: str) -> List[str]:
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(list(set(files)))

def load_audio(path: str, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)

def sample_window(audio: np.ndarray, sr: int, window_sec: float) -> np.ndarray:
    w = int(window_sec * sr)
    if audio.size < w:
        audio = np.pad(audio, (0, w-audio.size), mode="constant")
        return audio[:w]
    start = random.randint(0, max(0, audio.size - w))
    return audio[start:start+w]

@dataclass
class Item:
    path: str
    label: int

class AudioDataset:
    def __init__(self, split_dir: str, target_sr: int = 16000, window_sec: float = 15.0):
        self.target_sr = target_sr
        self.window_sec = window_sec
        self.items: List[Item] = []
        for i, lab in enumerate(LABELS):
            d = os.path.join(split_dir, lab)
            for p in list_audio_files(d):
                self.items.append(Item(path=p, label=i))
        if not self.items:
            raise RuntimeError(f"No hay audios en {split_dir}. Revisa tu dataset.")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        it = self.items[idx]
        audio = load_audio(it.path, self.target_sr)
        win = sample_window(audio, self.target_sr, self.window_sec)
        return win, it.label
