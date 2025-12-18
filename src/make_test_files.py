import os
import argparse
import numpy as np

def find_key(npz, preferred):
    for k in preferred:
        if k in npz.files:
            return k
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/processed/mitbih_windows.npz")
    ap.add_argument("--out", default="web/test_samples")
    ap.add_argument("--per_class", type=int, default=10, help="files per class")
    ap.add_argument("--random", type=int, default=20, help="random files")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)

    # Try common keys first; fallback to guessing
    X_key = find_key(d, ["X", "x", "signals", "windows"])
    y_key = find_key(d, ["y", "Y", "y_class", "labels"])
    hr_key = find_key(d, ["y_hr", "hr", "yHR", "heart_rate"])

    if X_key is None:
        # fallback: pick the first 3D array
        for k in d.files:
            if getattr(d[k], "ndim", 0) == 3:
                X_key = k
                break
    if X_key is None:
        raise RuntimeError(f"Could not find X in {d.files}")

    X = d[X_key]  # expected (N,1,L) or (N,L)
    if X.ndim == 3:
        X1 = X[:, 0, :]
    elif X.ndim == 2:
        X1 = X
    else:
        raise RuntimeError(f"Unexpected X shape: {X.shape}")

    N, L = X1.shape

    y = None
    if y_key is not None:
        y = d[y_key].astype(int).reshape(-1)
        if len(y) != N:
            print(f"⚠️ y length mismatch: {len(y)} vs N={N}. Ignoring y.")
            y = None

    y_hr = None
    if hr_key is not None:
        y_hr = d[hr_key].astype(float).reshape(-1)
        if len(y_hr) != N:
            print(f"⚠️ y_hr length mismatch: {len(y_hr)} vs N={N}. Ignoring y_hr.")
            y_hr = None

    rng = np.random.default_rng(args.seed)

    saved = []
    # 1) Random samples
    rand_idx = rng.choice(N, size=min(args.random, N), replace=False)
    for idx in rand_idx:
        label = int(y[idx]) if y is not None else -1
        hr = float(y_hr[idx]) if y_hr is not None else float("nan")
        sig = X1[idx].astype(np.float32)

        base = f"random_idx{idx}_label{label}"
        np.save(os.path.join(args.out, base + ".npy"), sig)
        np.savetxt(os.path.join(args.out, base + ".csv"), sig, delimiter=",")
        saved.append((base, label, hr, idx))

    # 2) Per-class samples (if labels exist)
    if y is not None:
        classes = sorted(set(int(v) for v in y.tolist()))
        for c in classes:
            idxs = np.where(y == c)[0]
            if len(idxs) == 0:
                continue
            pick = rng.choice(idxs, size=min(args.per_class, len(idxs)), replace=False)
            for idx in pick:
                label = int(y[idx])
                hr = float(y_hr[idx]) if y_hr is not None else float("nan")
                sig = X1[idx].astype(np.float32)

                base = f"class{label}_idx{idx}"
                np.save(os.path.join(args.out, base + ".npy"), sig)
                np.savetxt(os.path.join(args.out, base + ".csv"), sig, delimiter=",")
                saved.append((base, label, hr, idx))

    # Write an index file (so you know what each file is)
    index_path = os.path.join(args.out, "index.csv")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("file_base,label,hr_bpm,source_index\n")
        for base, label, hr, idx in saved:
            f.write(f"{base},{label},{hr},{idx}\n")

    print(f"✅ Loaded: {args.npz}")
    print(f"✅ X key: {X_key} | shape: {X.shape} -> using (N,L)=({N},{L})")
    print(f"✅ y key: {y_key} | hr key: {hr_key}")
    print(f"✅ Saved {len(saved)} windows to: {args.out}")
    print(f"✅ Index file: {index_path}")

if __name__ == "__main__":
    main()
