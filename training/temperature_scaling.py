from __future__ import annotations
import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

from training.dataset import AudioDataset, LABELS
from backend.app.model import build_model, encode_mean
from backend.app.infer import WEIGHTS_DIR

def collate(batch):
    xs, ys = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(()))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        t = torch.exp(self.log_t).clamp(0.05, 10.0)
        return logits / t

def nll_loss(logits, y):
    return torch.nn.functional.cross_entropy(logits, y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--window_sec", type=float, default=15.0)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    val_ds = AudioDataset(os.path.join(args.data_dir, "val"), target_sr=args.sample_rate, window_sec=args.window_sec)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    bundle = build_model(labels=LABELS)
    encoder = bundle.encoder.to(device).eval()
    head = bundle.head.to(device).eval()

    weights_path = os.path.join("backend","weights","model.pt")
    if not os.path.exists(weights_path):
        raise SystemExit("No existe backend/weights/model.pt. Entrena primero.")
    ckpt = torch.load(weights_path, map_location=device)
    head.load_state_dict(ckpt["head"])
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])

    # collect logits/labels
    all_logits = []
    all_y = []
    with torch.inference_mode():
        for x, y in tqdm(val_loader, desc="collect"):
            x = x.to(device)
            y = y.to(device)
            if args.sample_rate != bundle.encoder_sr:
                x = torchaudio.functional.resample(x, args.sample_rate, bundle.encoder_sr)
            emb = encode_mean(encoder, x)
            logits = head(emb)
            all_logits.append(logits)
            all_y.append(y)
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)

    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.5, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = nll_loss(scaler(logits), y)
        loss.backward()
        return loss

    opt.step(closure)

    t = float(torch.exp(scaler.log_t).detach().cpu().item())
    out_path = os.path.join("backend","weights","temperature.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"temperature": t}, f, indent=2)
    print("Saved temperature:", t, "->", out_path)

if __name__ == "__main__":
    main()
