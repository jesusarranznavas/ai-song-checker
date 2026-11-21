from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from training.dataset import AudioDataset, LABELS
from backend.app.model import build_model, encode_mean
from backend.app.calibrate import load_temperature, apply_temperature

def collate(batch):
    xs, ys = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--window_sec", type=float, default=15.0)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Labels:", LABELS)

    ds = AudioDataset(os.path.join(args.data_dir, args.split), target_sr=args.sample_rate, window_sec=args.window_sec)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )

    bundle = build_model(labels=LABELS)
    encoder = bundle.encoder.to(device).eval()
    head = bundle.head.to(device).eval()

    weights_path = os.path.join("backend","weights","model.pt")
    if not os.path.exists(weights_path):
        raise SystemExit("Falta ../backend/weights/model.pt. Entrena primero.")
    ckpt = torch.load(weights_path, map_location=device)
    head.load_state_dict(ckpt["head"])
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])

    temp_path = os.path.join("backend","weights","temperature.json")
    temp = load_temperature(temp_path) if os.path.exists(temp_path) else None
    if temp is not None:
        print("Temperature:", temp)
    else:
        print("Temperature: (none)")

    num_classes = len(LABELS)
    conf = np.zeros((num_classes, num_classes), dtype=int)

    correct = 0
    n = 0
    with torch.inference_mode():
        for x, y in tqdm(loader, desc=f"eval/{args.split}"):
            x = x.to(device)
            y = y.to(device)

            if args.sample_rate != bundle.encoder_sr:
                x = torchaudio.functional.resample(x, args.sample_rate, bundle.encoder_sr)

            emb = encode_mean(encoder, x)
            logits = head(emb)
            logits = apply_temperature(logits, temp)

            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            n += y.numel()

            for pi, yi in zip(pred.cpu().numpy(), y.cpu().numpy()):
                conf[int(yi), int(pi)] += 1

    acc = correct / max(1, n)
    print("Accuracy:", acc)
    print("Confusion (rows=true, cols=pred):\n", conf)

    # métricas simples por clase
    for i, lab in enumerate(LABELS):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = (2 * prec * rec) / max(1e-9, (prec + rec))
        print(f"{lab}: precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")

if __name__ == "__main__":
    main()
