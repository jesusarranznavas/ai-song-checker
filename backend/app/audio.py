import io
import os
import tempfile
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa
import ffmpeg

def _decode_with_ffmpeg(file_bytes: bytes, target_sr: int) -> Tuple[np.ndarray, int]:
    # Decodifica cualquier formato soportado por ffmpeg -> wav float32 mono
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
        tmp_in.write(file_bytes)
        in_path = tmp_in.name

    out_path = in_path + ".wav"
    try:
        (
            ffmpeg
            .input(in_path)
            .output(out_path, format="wav", acodec="pcm_f32le", ac=1, ar=target_sr)
            .overwrite_output()
            .run(quiet=True)
        )
        audio, sr = sf.read(out_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr
    finally:
        for p in [in_path, out_path]:
            try:
                os.remove(p)
            except Exception:
                pass

def load_audio_bytes(file_bytes: bytes, target_sr: int) -> np.ndarray:
    try:
        audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)
    except Exception:
        audio, sr = _decode_with_ffmpeg(file_bytes, target_sr)
        return audio.astype(np.float32)

def window_audio(audio: np.ndarray, sr: int, window_sec: float, stride_sec: float, max_windows: int) -> List[np.ndarray]:
    w = int(window_sec * sr)
    s = int(stride_sec * sr)
    if w <= 0 or s <= 0:
        raise ValueError("window_sec/stride_sec inválidos")
    if audio.size < w:
        # pad
        pad = w - audio.size
        audio = np.pad(audio, (0, pad), mode="constant")
    windows = []
    start = 0
    while start + w <= audio.size and len(windows) < max_windows:
        windows.append(audio[start:start+w])
        start += s
    if not windows:
        windows = [audio[:w]]
    return windows
