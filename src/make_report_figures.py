#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


def set_style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_dir / (name + '.png')}")


def plot_training_curves(history_csv: Path, out_dir: Path):
    if not history_csv.exists():
        print(f"[SKIP] Missing history: {history_csv}")
        return

    import pandas as pd
    df = pd.read_csv(history_csv)

    if "epoch" not in df.columns:
        df["epoch"] = np.arange(1, len(df) + 1)

    epochs = df["epoch"].values

    # Loss
    if "train_loss" in df.columns or "val_loss" in df.columns:
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        if "train_loss" in df.columns:
            ax.plot(epochs, df["train_loss"].values, label="train loss")
        if "val_loss" in df.columns:
            ax.plot(epochs, df["val_loss"].values, label="val loss")
        ax.set_title("Training Curves — Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        save(fig, out_dir, "fig_training_loss")

    # Accuracy
    if "train_acc" in df.columns or "val_acc" in df.columns:
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        if "train_acc" in df.columns:
            ax.plot(epochs, df["train_acc"].values, label="train acc")
        if "val_acc" in df.columns:
            ax.plot(epochs, df["val_acc"].values, label="val acc")
        ax.set_title("Training Curves — Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        save(fig, out_dir, "fig_training_accuracy")


def plot_confusion_matrix(test_npz: Path, out_dir: Path, normalize=True):
    if not test_npz.exists():
        print(f"[SKIP] Missing test outputs: {test_npz}")
        return

    d = np.load(test_npz, allow_pickle=True)
    if "y_true" not in d.files or "y_pred" not in d.files:
        print(f"[SKIP] test_outputs.npz must contain y_true & y_pred. Found: {d.files}")
        return

    y_true = d["y_true"].astype(int)
    y_pred = d["y_pred"].astype(int)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float)
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row != 0)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([str(i) for i in labels], rotation=35, ha="right")
    ax.set_yticklabels([str(i) for i in labels])

    fmt = ".2f" if normalize else "d"
    thresh = (cm.max() + cm.min()) / 2.0 if np.isfinite(cm).all() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, format(v, fmt),
                    ha="center", va="center", fontsize=8,
                    color="white" if v > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.grid(False)
    save(fig, out_dir, "fig_confusion_matrix")


def plot_roc_pr(test_npz: Path, out_dir: Path):
    if not test_npz.exists():
        print(f"[SKIP] Missing test outputs: {test_npz}")
        return

    d = np.load(test_npz, allow_pickle=True)
    if "y_true" not in d.files or "y_prob" not in d.files:
        print(f"[SKIP] ROC/PR needs y_true + y_prob in npz. Found: {d.files}")
        return

    y_true = d["y_true"].astype(int)
    y_prob = d["y_prob"].astype(float)

    if y_prob.ndim != 2:
        print(f"[SKIP] y_prob must be shape (N,C). Got: {y_prob.shape}")
        return

    n_classes = y_prob.shape[1]
    classes = np.arange(n_classes)
    y_bin = label_binarize(y_true, classes=classes)

    # ROC
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        ax.plot(fpr, tpr, label=f"class {c} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="chance")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(ncol=2)
    save(fig, out_dir, "fig_roc_curves")

    # PR
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for c in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, c], y_prob[:, c])
        ap = average_precision_score(y_bin[:, c], y_prob[:, c])
        ax.plot(rec, prec, label=f"class {c} (AP={ap:.3f})")
    ax.set_title("Precision–Recall Curves (One-vs-Rest)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(ncol=2)
    save(fig, out_dir, "fig_pr_curves")


def to_NTC(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 2:
        return X[:, :, None]
    if X.ndim != 3:
        raise ValueError(f"Expected 2D/3D windows, got {X.shape}")
    N, d1, d2 = X.shape
    # if last dim is bigger -> likely (N,C,T)
    if d2 > d1:
        return np.transpose(X, (0, 2, 1))
    return X


def plot_error_analysis(windows_npz: Path, test_npz: Path, out_dir: Path, k_each: int = 3, seed: int = 0):
    if not windows_npz.exists():
        print(f"[SKIP] Missing windows npz: {windows_npz}")
        return
    if not test_npz.exists():
        print(f"[SKIP] Missing test outputs: {test_npz}")
        return

    W = np.load(windows_npz, allow_pickle=True)
    D = np.load(test_npz, allow_pickle=True)

    # locate windows key
    win_key = None
    for cand in ["X", "x", "windows", "signals", "data"]:
        if cand in W.files:
            win_key = cand
            break
    if win_key is None:
        print(f"[SKIP] Could not find windows array. Keys: {W.files}")
        return

    X = to_NTC(W[win_key])  # (N,T,C)
    y_true = D["y_true"].astype(int)
    y_pred = D["y_pred"].astype(int)

    n = min(len(y_true), X.shape[0])
    X = X[:n]
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx = np.where(y_true != y_pred)[0]

    rng = np.random.default_rng(seed)
    pick_c = rng.choice(correct_idx, size=min(k_each, len(correct_idx)), replace=False) if len(correct_idx) else np.array([], dtype=int)
    pick_w = rng.choice(wrong_idx, size=min(k_each, len(wrong_idx)), replace=False) if len(wrong_idx) else np.array([], dtype=int)

    picks = [("Correct", i) for i in pick_c] + [("Wrong", i) for i in pick_w]
    if len(picks) == 0:
        print("[SKIP] No samples available for error analysis.")
        return

    rows = len(picks)
    fig, axes = plt.subplots(rows, 1, figsize=(7.2, 1.4 * rows), sharex=False)
    if rows == 1:
        axes = [axes]

    for ax, (tag, i) in zip(axes, picks):
        ax.plot(X[i, :, 0], linewidth=0.9)
        ax.set_title(f"{tag} — idx={i} | true={y_true[i]} pred={y_pred[i]}")
        ax.set_xlabel("sample index")
        ax.set_ylabel("amp")

    save(fig, out_dir, "fig_error_analysis_examples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history_csv", default=r"results\figures\history.csv")
    ap.add_argument("--test_npz", default=r"results\figures\test_outputs.npz")
    ap.add_argument("--windows_npz", default=r"data\processed\mitbih_windows.npz")
    ap.add_argument("--fig_dir", default=r"results\figures")
    args = ap.parse_args()

    set_style()
    out_dir = Path(args.fig_dir).resolve()

    plot_training_curves(Path(args.history_csv).resolve(), out_dir)
    plot_confusion_matrix(Path(args.test_npz).resolve(), out_dir, normalize=True)
    plot_roc_pr(Path(args.test_npz).resolve(), out_dir)
    plot_error_analysis(Path(args.windows_npz).resolve(), Path(args.test_npz).resolve(), out_dir, k_each=3, seed=0)


if __name__ == "__main__":
    main()
