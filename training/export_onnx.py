from __future__ import annotations
import os
import argparse
import torch

from backend.app.model import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=os.path.join("..","backend","weights","head.onnx"))
    args = ap.parse_args()

    device = "cpu"
    bundle = build_model()
    head = bundle.head.to(device).eval()

    ckpt_path = os.path.join("..","backend","weights","model.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit("Entrena primero: falta backend/weights/model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(ckpt["head"])

    dummy = torch.randn(1, 768)  # dim típica de wav2vec2 base
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(head, dummy, args.out, input_names=["emb"], output_names=["logits"], opset_version=17)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
