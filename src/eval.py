from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def save_test_outputs(
    out_path: str,
    y_true,
    y_pred,
    y_prob: Optional[np.ndarray] = None,
    y_true_hr: Optional[np.ndarray] = None,
    y_pred_hr: Optional[np.ndarray] = None,
) -> None:
    """
    Save evaluation outputs for plotting:
      - y_true, y_pred (classification)
      - y_prob (softmax probabilities, optional but needed for ROC/PR)
      - y_true_hr, y_pred_hr (optional regression)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    }

    if y_prob is not None:
        payload["y_prob"] = np.asarray(y_prob)

    if y_true_hr is not None:
        payload["y_true_hr"] = np.asarray(y_true_hr)

    if y_pred_hr is not None:
        payload["y_pred_hr"] = np.asarray(y_pred_hr)

    np.savez(out_path, **payload)
    print(f"[SAVED] {out_path}")


def _get_base_dataset(ds):
    """
    If loader.dataset is a torch.utils.data.Subset, unwrap it to get the real dataset.
    """
    if hasattr(ds, "dataset"):
        return ds.dataset
    return ds


@torch.no_grad()
def evaluate(model, loader, device, save_npz: bool = True) -> Tuple[str, np.ndarray, float, float]:
    """
    Runs evaluation and returns:
      report (str), confusion matrix (np.ndarray), mae (float), rmse (float)

    Also saves results/figures/test_outputs.npz by default.
    """
    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    hr_true = []
    hr_pred = []

    for batch in loader:
        # Your loader yields: (x, y_class, y_hr)
        x, y_class, y_hr = batch

        x = x.to(device)

        # model returns: logits, hr_hat, _  (based on your code)
        logits, hr_hat, _ = model(x)

        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(logits, dim=1)

        all_true.append(y_class.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(probs.cpu().numpy())

        hr_true.append(y_hr.cpu().numpy())
        hr_pred.append(hr_hat.cpu().numpy())

    # concat
    y_true = np.concatenate(all_true).astype(int)
    y_pred = np.concatenate(all_pred).astype(int)
    y_prob = np.concatenate(all_prob).astype(float)

    hr_true = np.concatenate(hr_true).reshape(-1).astype(float)
    hr_pred = np.concatenate(hr_pred).reshape(-1).astype(float)

    # Denormalize HR if dataset stored mean/std
    ds_obj = _get_base_dataset(loader.dataset)
    if hasattr(ds_obj, "hr_mean") and hasattr(ds_obj, "hr_std"):
        hr_mean = float(ds_obj.hr_mean)
        hr_std = float(ds_obj.hr_std)
        hr_true = hr_true * hr_std + hr_mean
        hr_pred = hr_pred * hr_std + hr_mean

    # HR metrics
    mae = float(np.mean(np.abs(hr_true - hr_pred)))
    rmse = float(np.sqrt(np.mean((hr_true - hr_pred) ** 2)))

    # Classification metrics
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    print("Normalized CM:\n", np.round(cm_norm, 3))
    print(f"HR MAE: {mae:.3f} | HR RMSE: {rmse:.3f}")

    # Save outputs for plotting
    if save_npz:
        save_test_outputs(
            out_path=r"results\figures\test_outputs.npz",
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,            # needed for ROC/PR
            y_true_hr=hr_true,        # optional (for HR scatter/bland-altman)
            y_pred_hr=hr_pred,
        )

    return report, cm, mae, rmse
