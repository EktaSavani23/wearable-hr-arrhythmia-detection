# src/print_class_distribution.py
import numpy as np
import torch

from src.config import Config
from src.dataset import WearableMultiTaskDataset

def count_labels(ds, key_candidates=("y_cls", "y", "label", "labels")):
    """
    Tries to extract the class label from each sample.
    Works even if __getitem__ returns dict/tuple.
    """
    counts = None
    for i in range(len(ds)):
        item = ds[i]

        # dict-style return
        if isinstance(item, dict):
            y = None
            for k in key_candidates:
                if k in item:
                    y = item[k]
                    break
            if y is None:
                raise KeyError(f"Could not find label key in dict sample. Keys: {list(item.keys())}")
        else:
            # tuple/list return: assume (x, y) or (x, y, ...)
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                y = item[1]
            else:
                raise TypeError(f"Unexpected sample type: {type(item)}")

        # convert to int
        if torch.is_tensor(y):
            y = int(y.item())
        else:
            y = int(y)

        if counts is None:
            # try to use Config.num_classes if available
            num_classes = getattr(Config(), "num_classes", None)
            if num_classes is None:
                num_classes = 5  # fallback
            counts = np.zeros(num_classes, dtype=int)

        if y >= len(counts):
            # expand if needed
            new_counts = np.zeros(y + 1, dtype=int)
            new_counts[: len(counts)] = counts
            counts = new_counts

        counts[y] += 1

    return counts

def main():
    cfg = Config()

    # You might have different split names/paths; adjust if needed:
    train_ds = WearableMultiTaskDataset(cfg, split="train")
    val_ds   = WearableMultiTaskDataset(cfg, split="val")
    test_ds  = WearableMultiTaskDataset(cfg, split="test")

    train_c = count_labels(train_ds)
    val_c   = count_labels(val_ds)
    test_c  = count_labels(test_ds)

    print("\nCounts per class index:")
    print("Train:", train_c)
    print("Val:  ", val_c)
    print("Test: ", test_c)

    # Optional: show totals
    print("\nTotals:")
    print("Train total:", int(train_c.sum()))
    print("Val total:  ", int(val_c.sum()))
    print("Test total: ", int(test_c.sum()))

    # Optional mapping (edit if your mapping differs)
    class_names = ["N", "S", "V", "F", "Q"]
    if len(train_c) == len(class_names):
        print("\nCounts with AAMI labels:")
        for i, name in enumerate(class_names):
            print(f"{name}: Train={train_c[i]}  Val={val_c[i]}  Test={test_c[i]}")

if __name__ == "__main__":
    main()
