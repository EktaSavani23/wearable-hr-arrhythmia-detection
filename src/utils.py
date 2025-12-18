import os
import random
import numpy as np
import torch

def ensure_dirs():
    os.makedirs("data/raw_mitbih", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(use_cuda: bool = True) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint(path: str, model, optimizer, epoch: int, best_val: float):
    torch.save({
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
    }, path)

def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["opt_state"])
    return ckpt
