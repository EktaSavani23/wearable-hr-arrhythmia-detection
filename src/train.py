from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# -----------------------------
# CSV logging (for report figs)
# -----------------------------
def append_history_row(csv_path: str, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------
# Loss: Cross-entropy with label smoothing
# -----------------------------
def ce_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Cross entropy with optional label smoothing and optional class weights.
    targets shape: (B,)
    logits shape: (B,K)
    class_weights shape: (K,)
    """
    if smoothing <= 0.0:
        return F.cross_entropy(logits, targets, weight=class_weights)

    K = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)

    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (K - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)

    # weighted smoothed CE
    if class_weights is not None:
        w = class_weights.unsqueeze(0)  # (1,K)
        loss = (-true_dist * log_probs) * w
        return loss.sum(dim=1).mean()

    return (-true_dist * log_probs).sum(dim=1).mean()


# -----------------------------
# Metrics helpers
# -----------------------------
def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    correct = (pred == y).sum().item()
    return correct / max(1, y.numel())


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return mae, rmse


def _unwrap_dataset(ds):
    # if Subset, unwrap
    return ds.dataset if hasattr(ds, "dataset") else ds


# -----------------------------
# One epoch: train
# -----------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    lambda_hr: float = 0.15,
    class_weights: Optional[torch.Tensor] = None,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.05,
) -> Dict[str, float]:
    model.train()
    hr_loss_fn = nn.SmoothL1Loss().to(device)

    sum_total = 0.0
    sum_cls = 0.0
    sum_hr = 0.0
    sum_acc = 0.0
    n_samples = 0

    for x, y_class, y_hr in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y_class = y_class.to(device)
        y_hr = y_hr.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, hr_hat, _ = model(x)

        loss_cls = ce_label_smoothing(logits, y_class, class_weights, label_smoothing)
        loss_hr = hr_loss_fn(hr_hat, y_hr)
        loss = loss_cls + lambda_hr * loss_hr

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = x.size(0)
        n_samples += bs
        sum_total += loss.item() * bs
        sum_cls += loss_cls.item() * bs
        sum_hr += loss_hr.item() * bs
        sum_acc += _accuracy_from_logits(logits, y_class) * bs

    return {
        "train_total_loss": sum_total / max(1, n_samples),
        "train_cls_loss": sum_cls / max(1, n_samples),
        "train_hr_loss": sum_hr / max(1, n_samples),
        "train_acc": sum_acc / max(1, n_samples),
    }


# -----------------------------
# One epoch: validation
# -----------------------------
@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    lambda_hr: float = 0.15,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.05,
    denorm_hr: bool = True,
) -> Dict[str, float]:
    model.eval()
    hr_loss_fn = nn.SmoothL1Loss().to(device)

    sum_total = 0.0
    sum_cls = 0.0
    sum_hr = 0.0
    sum_acc = 0.0
    n_samples = 0

    # for HR MAE/RMSE
    hr_true_all = []
    hr_pred_all = []

    ds_obj = _unwrap_dataset(loader.dataset)
    hr_mean = float(getattr(ds_obj, "hr_mean", 0.0))
    hr_std = float(getattr(ds_obj, "hr_std", 1.0))

    for x, y_class, y_hr in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        y_class = y_class.to(device)
        y_hr = y_hr.to(device)

        logits, hr_hat, _ = model(x)

        loss_cls = ce_label_smoothing(logits, y_class, class_weights, label_smoothing)
        loss_hr = hr_loss_fn(hr_hat, y_hr)
        loss = loss_cls + lambda_hr * loss_hr

        bs = x.size(0)
        n_samples += bs
        sum_total += loss.item() * bs
        sum_cls += loss_cls.item() * bs
        sum_hr += loss_hr.item() * bs
        sum_acc += _accuracy_from_logits(logits, y_class) * bs

        # HR arrays (cpu)
        hr_true_all.append(y_hr.detach().cpu().numpy().reshape(-1))
        hr_pred_all.append(hr_hat.detach().cpu().numpy().reshape(-1))

    hr_true = np.concatenate(hr_true_all) if hr_true_all else np.array([])
    hr_pred = np.concatenate(hr_pred_all) if hr_pred_all else np.array([])

    if denorm_hr and hr_true.size > 0:
        # if dataset uses normalization
        if hasattr(ds_obj, "hr_mean") and hasattr(ds_obj, "hr_std"):
            hr_true = hr_true * hr_std + hr_mean
            hr_pred = hr_pred * hr_std + hr_mean

    hr_mae, hr_rmse = (np.nan, np.nan)
    if hr_true.size > 0:
        hr_mae, hr_rmse = _mae_rmse(hr_true, hr_pred)

    return {
        "val_total_loss": sum_total / max(1, n_samples),
        "val_cls_loss": sum_cls / max(1, n_samples),
        "val_hr_loss": sum_hr / max(1, n_samples),
        "val_acc": sum_acc / max(1, n_samples),
        "val_hr_mae": float(hr_mae),
        "val_hr_rmse": float(hr_rmse),
    }


# -----------------------------
# Fit loop (call this from main_train.py or train entry)
# -----------------------------
def fit(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epochs: int = 20,
    lambda_hr: float = 0.15,
    class_weights: Optional[torch.Tensor] = None,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.05,
    history_csv_path: str = r"results\figures\history.csv",
) -> None:
    """
    Train + validate for `epochs` and write a CSV row per epoch (for plotting curves).
    """
    for epoch in range(epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_hr=lambda_hr,
            class_weights=class_weights,
            grad_clip=grad_clip,
            label_smoothing=label_smoothing,
        )

        val_stats = validate_one_epoch(
            model,
            val_loader,
            device,
            lambda_hr=lambda_hr,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            denorm_hr=True,
        )

        row = {
            "epoch": epoch + 1,

            # total losses
            "train_loss": train_stats["train_total_loss"],
            "val_loss": val_stats["val_total_loss"],

            # classification-only losses
            "train_cls_loss": train_stats["train_cls_loss"],
            "val_cls_loss": val_stats["val_cls_loss"],

            # hr losses
            "train_hr_loss": train_stats["train_hr_loss"],
            "val_hr_loss": val_stats["val_hr_loss"],

            # accuracy
            "train_acc": train_stats["train_acc"],
            "val_acc": val_stats["val_acc"],

            # HR metrics (val only)
            "val_hr_mae": val_stats["val_hr_mae"],
            "val_hr_rmse": val_stats["val_hr_rmse"],
        }

        append_history_row(history_csv_path, row)

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"train loss={row['train_loss']:.4f}, acc={row['train_acc']:.4f} | "
            f"val loss={row['val_loss']:.4f}, acc={row['val_acc']:.4f} | "
            f"val HR MAE={row['val_hr_mae']:.3f}, RMSE={row['val_hr_rmse']:.3f}"
        )
