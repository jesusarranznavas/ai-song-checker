from __future__ import annotations
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from training.dataset import AudioDataset, LABELS
from training.augment import augment

import torchaudio
from backend.app.model import build_model, encode_mean


def collate(batch):
    xs, ys = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()  # [B, T]
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="ruta a ./data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--window_sec", type=float, default=15.0)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--fine_tune_encoder", action="store_true", help="afinar encoder (más VRAM)")
    ap.add_argument("--encoder_lr", type=float, default=1e-5, help="lr para el encoder cuando se afina")
    ap.add_argument("--encoder_weights", type=str, default=os.path.join("backend", "weights", "model.pt"),
                    help="pesos iniciales del encoder MERT (checkpoint del proyecto o pytorch_model.bin de MERT)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Labels:", LABELS)

    train_ds = AudioDataset(os.path.join(args.data_dir, "train"), target_sr=args.sample_rate, window_sec=args.window_sec)
    val_ds = AudioDataset(os.path.join(args.data_dir, "val"), target_sr=args.sample_rate, window_sec=args.window_sec)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )

    if not os.path.exists(args.encoder_weights):
        raise SystemExit(f"No existe {args.encoder_weights}. Ejecuta: python scripts/download_weights.py")
    bundle = build_model(labels=LABELS, encoder_weights=args.encoder_weights)
    encoder = bundle.encoder.to(device).eval()
    head = bundle.head.to(device).train()

    if args.fine_tune_encoder:
        encoder.train()

    if args.fine_tune_encoder:
        params = [
            {"params": head.parameters(), "lr": args.lr},
            {"params": encoder.parameters(), "lr": args.encoder_lr},
        ]
    else:
        params = list(head.parameters())
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        head.train()
        if args.fine_tune_encoder:
            encoder.train()
        else:
            encoder.eval()

        pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")
        total_loss = 0.0

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            # augment per sample
            x_aug = []
            for i in range(x.shape[0]):
                wav = x[i].unsqueeze(0)  # [1, T]
                wav = augment(wav, args.sample_rate)
                x_aug.append(wav)
            x = torch.cat(x_aug, dim=0)  # [B, T]

            # resample to encoder sr if needed
            if args.sample_rate != bundle.encoder_sr:
                x = torchaudio.functional.resample(x, args.sample_rate, bundle.encoder_sr)

            emb = encode_mean(encoder, x)
            logits = head(emb)
            loss = ce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=total_loss / max(1, len(pbar)))

        # val
        head.eval()
        encoder.eval()
        correct = 0
        n = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                if args.sample_rate != bundle.encoder_sr:
                    x = torchaudio.functional.resample(x, args.sample_rate, bundle.encoder_sr)
                emb = encode_mean(encoder, x)
                logits = head(emb)
                pred = logits.argmax(dim=-1)
                correct += int((pred == y).sum().item())
                n += y.numel()

        acc = correct / max(1, n)
        scheduler.step()
        print(f"Epoch {epoch}: val_acc={acc:.4f}")

        if acc > best:
            best = acc
            os.makedirs(os.path.join("backend", "weights"), exist_ok=True)
            ckpt = {"head": head.state_dict()}
            if args.fine_tune_encoder:
                ckpt["encoder"] = encoder.state_dict()
            torch.save(ckpt, os.path.join("backend", "weights", "model.pt"))
            print("Saved best -> backend/weights/model.pt")

    print("Done. Best val_acc:", best)


if __name__ == "__main__":
    main()
