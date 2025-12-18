from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.config import Config
from src.utils import ensure_dirs, set_seed, get_device, save_checkpoint
from src.preprocess_mitbih import save_processed
from src.dataset import WearableMultiTaskDataset
from src.model import FastMultiTaskCNN
from src.train import train_one_epoch, validate_one_epoch, append_history_row
from src.eval import evaluate


def compute_mild_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Mild inverse-frequency weights, clipped, mean-normalized.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w = 1.0 / np.sqrt(counts)           # mild weighting
    w = w / w.mean()                    # normalize mean=1
    w = np.clip(w, 0.7, 2.5)            # clip to avoid domination
    return torch.tensor(w, dtype=torch.float32)


def main() -> None:
    cfg = Config()
    ensure_dirs()
    set_seed(cfg.seed)

    device = get_device(cfg.use_cuda)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 1) Preprocess if needed
    if not os.path.exists(cfg.processed_path):
        print("Processed file not found. Creating windows + labels...")
        save_processed(cfg)

    # 2) Load dataset
    ds = WearableMultiTaskDataset(cfg.processed_path)

    # 3) Split train/val/test
    n = len(ds)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 4) Compute class weights from training subset
    y_train = np.array([train_ds[i][1].item() for i in range(len(train_ds))], dtype=np.int64)
    class_weights = compute_mild_class_weights(y_train, cfg.num_classes).to(device)

    # Optional tweak you had (only if class 2 exists)
    if cfg.num_classes > 2:
        class_weights[2] = class_weights[2] * 1.5

    # 5) Build model + optimizer
    model = FastMultiTaskCNN(num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # 6) Scheduler (optional)
    scheduler = None
    if getattr(cfg, "use_scheduler", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=getattr(cfg, "scheduler_factor", 0.5),
            patience=getattr(cfg, "scheduler_patience", 2),
        )

    # 7) Training loop (writes history.csv for report figures)
    history_csv = r"results\figures\history.csv"
    best_val = float("inf")

    print("\n=== Training start ===")
    for epoch in range(1, cfg.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_hr=cfg.lambda_hr,
            class_weights=class_weights,
            grad_clip=getattr(cfg, "grad_clip", 1.0),
            label_smoothing=cfg.label_smoothing,
        )

        val_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            lambda_hr=cfg.lambda_hr,
            class_weights=class_weights,
            label_smoothing=cfg.label_smoothing,
            denorm_hr=True,
        )

        # Save a row for plotting training curves
        row = {
            "epoch": epoch,
            "lr": float(lr_now),

            "train_loss": float(train_stats["train_total_loss"]),
            "val_loss": float(val_stats["val_total_loss"]),

            "train_cls_loss": float(train_stats["train_cls_loss"]),
            "val_cls_loss": float(val_stats["val_cls_loss"]),

            "train_hr_loss": float(train_stats["train_hr_loss"]),
            "val_hr_loss": float(val_stats["val_hr_loss"]),

            "train_acc": float(train_stats["train_acc"]),
            "val_acc": float(val_stats["val_acc"]),

            "val_hr_mae": float(val_stats["val_hr_mae"]),
            "val_hr_rmse": float(val_stats["val_hr_rmse"]),
        }
        append_history_row(history_csv, row)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | lr={lr_now:.2e} | "
            f"train loss={row['train_loss']:.4f} acc={row['train_acc']:.4f} | "
            f"val loss={row['val_loss']:.4f} acc={row['val_acc']:.4f} | "
            f"val HR MAE={row['val_hr_mae']:.3f} RMSE={row['val_hr_rmse']:.3f}"
        )

        # Step scheduler on validation total loss
        if scheduler is not None:
            scheduler.step(row["val_loss"])

        # Save best checkpoint
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            save_checkpoint("results/checkpoints/best.pt", model, optimizer, epoch, best_val)
            print("✅ Saved BEST checkpoint.")

        # Save last checkpoint
        save_checkpoint("results/checkpoints/last.pt", model, optimizer, epoch, best_val)

    print("=== Training complete ===\n")

    # 8) Evaluate best model on test set (eval.py saves test_outputs.npz automatically)
    ckpt = torch.load("results/checkpoints/best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    report, cm, mae, rmse = evaluate(model, test_loader, device, save_npz=True)
    print("\n=== Classification Report ===\n", report)
    print("Confusion Matrix:\n", cm)
    print(f"HR MAE: {mae:.3f} bpm | HR RMSE: {rmse:.3f} bpm")

    print("\n✅ Saved:")
    print(" - results/figures/history.csv (training curves)")
    print(" - results/figures/test_outputs.npz (CM/ROC/PR + error analysis)")


if __name__ == "__main__":
    main()
