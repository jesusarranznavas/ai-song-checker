"""
Baseline classifiers (SVM + MFCC, Random Forest + MFCC) for comparison
with the MERT+MLP system in the thesis results table.

Run from the project root:
    python -m training.baseline --data_dir data
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from training.dataset import LABELS, list_audio_files

SR = 16000
WINDOW_SEC = 15.0
N_MFCC = 40


def extract_mfcc(path: str) -> np.ndarray:
    """Load audio and return mean+std of MFCCs over time (80-dim vector)."""
    try:
        audio, sr = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return None
    w = int(WINDOW_SEC * SR)
    if len(audio) < w:
        audio = np.pad(audio, (0, w - len(audio)))
    # take centre window to avoid intros/outros
    start = max(0, (len(audio) - w) // 2)
    audio = audio[start: start + w]
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def load_split(split_dir: str):
    X, y = [], []
    for i, lab in enumerate(LABELS):
        d = os.path.join(split_dir, lab)
        for p in list_audio_files(d):
            feat = extract_mfcc(p)
            if feat is not None:
                X.append(feat)
                y.append(i)
    return np.array(X), np.array(y)


def evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred):")
    print(f"              pred_human  pred_ai")
    print(f"  true_human      {cm[0,0]:<8}    {cm[0,1]}")
    print(f"  true_ai         {cm[1,0]:<8}    {cm[1,1]}")
    for i, lab in enumerate(LABELS):
        print(f"{lab}: precision={prec[i]:.3f}  recall={rec[i]:.3f}  f1={f1[i]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    args = ap.parse_args()

    print("Loading train split...")
    X_train, y_train = load_split(os.path.join(args.data_dir, "train"))
    print(f"  {len(X_train)} samples")

    print("Loading test split...")
    X_test, y_test = load_split(os.path.join(args.data_dir, "test"))
    print(f"  {len(X_test)} samples")

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
    ])
    svm.fit(X_train, y_train)
    evaluate("SVM + MFCC (baseline)", svm, X_test, y_test)

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    rf.fit(X_train, y_train)
    evaluate("Random Forest + MFCC (baseline)", rf, X_test, y_test)

    print("\n" + "="*50)
    print("  MERT + MLP (tu sistema) — resultados del train_log.txt")
    print("="*50)
    print("Accuracy: 0.8636")
    print("Confusion matrix:")
    print("              pred_human  pred_ai")
    print("  true_human      8           1")
    print("  true_ai         2          11")
    print("human:    precision=0.800  recall=0.889  f1=0.842")
    print("pure_ai:  precision=0.917  recall=0.846  f1=0.880")


if __name__ == "__main__":
    main()
