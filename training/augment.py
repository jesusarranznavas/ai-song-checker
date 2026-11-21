from __future__ import annotations
import random
import torch
import torchaudio

def random_gain(wav: torch.Tensor) -> torch.Tensor:
    # gain in dB
    gain_db = random.uniform(-6, 6)
    return wav * (10 ** (gain_db / 20))

def random_resample(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    # resample to a random sr then back (simula pipelines)
    target = random.choice([8000, 12000, 16000, 22050, 24000, 32000, 44100])
    if target == sr:
        return wav, sr
    wav2 = torchaudio.functional.resample(wav, sr, target)
    return wav2, target

def add_noise(wav: torch.Tensor) -> torch.Tensor:
    snr_db = random.uniform(15, 35)
    sig_power = wav.pow(2).mean().clamp_min(1e-9)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(wav) * noise_power.sqrt()
    return wav + noise

def augment(wav: torch.Tensor, sr: int) -> torch.Tensor:
    # wav: [1, T]
    if random.random() < 0.7:
        wav = random_gain(wav)
    if random.random() < 0.5:
        wav = add_noise(wav)
    # resample jitter (y vuelta) para robustez
    if random.random() < 0.3:
        wav2, sr2 = random_resample(wav, sr)
        wav = torchaudio.functional.resample(wav2, sr2, sr)
    wav = torch.clamp(wav, -1.0, 1.0)
    return wav
