import os
import numpy as np
import pandas as pd

NPZ_PATH = "data/processed/mitbih_windows.npz"
OUT_DIR  = "binary_test_samples"   # output folder
N_NORMAL = 50                      # how many "no arrhythmia" files
N_ARR    = 50                      # how many "has arrhythmia" files
SEED     = 0

# Assumption for MIT-BIH windows in your project:
# label 0 = Normal, label != 0 = Arrhythmia
NORMAL_LABEL = 0

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    d = np.load(NPZ_PATH, allow_pickle=True)

    # Your log confirmed these keys exist:
    X = d["X"]         # (N,1,L)
    y = d["y_class"]   # (N,)
    hr = d["y_hr"]     # (N,)

    X = X[:, 0, :]     # (N,L)
    N, L = X.shape

    rng = np.random.default_rng(SEED)

    normal_idx = np.where(y == NORMAL_LABEL)[0]
    arr_idx    = np.where(y != NORMAL_LABEL)[0]

    pick_normal = rng.choice(normal_idx, size=min(N_NORMAL, len(normal_idx)), replace=False)
    pick_arr    = rng.choice(arr_idx,    size=min(N_ARR,    len(arr_idx)),    replace=False)

    rows = []

    def save_one(idx, group_name):
        sig = X[idx].astype(np.float32).reshape(-1)
        label = int(y[idx])
        hr_bpm = float(hr[idx])
        has_arr = 0 if label == NORMAL_LABEL else 1

        file_base = f"{group_name}_idx{idx}_label{label}_arr{has_arr}"
        csv_path = os.path.join(OUT_DIR, file_base + ".csv")
        npy_path = os.path.join(OUT_DIR, file_base + ".npy")

        # Save signal only (what your UI expects)
        np.savetxt(csv_path, sig, delimiter=",")
        np.save(npy_path, sig)

        rows.append({
            "file": file_base,
            "group": group_name,
            "source_index": int(idx),
            "label_original": label,
            "has_arrhythmia": has_arr,
            "hr_bpm": hr_bpm,
            "length": int(len(sig)),
            "csv_path": csv_path,
            "npy_path": npy_path
        })

    for idx in pick_normal:
        save_one(int(idx), "normal")

    for idx in pick_arr:
        save_one(int(idx), "arrhythmia")

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    print("✅ Done.")
    print(f"✅ Saved files in: {OUT_DIR}")
    print(f"✅ Manifest (ground truth) saved to: {manifest_path}")
    print(f"✅ Created {len(pick_normal)} normal + {len(pick_arr)} arrhythmia files")
    print(f"✅ Each signal length = {L} samples")

if __name__ == "__main__":
    main()
