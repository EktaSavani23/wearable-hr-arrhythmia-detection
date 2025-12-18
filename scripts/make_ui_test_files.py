import os
import numpy as np
import pandas as pd
from pathlib import Path

# Always resolve paths relative to the project root (.../code)
BASE_DIR = Path(__file__).resolve().parents[1]  # E:\ekta_dla\code

PROCESSED_NPZ = BASE_DIR / "data" / "processed" / "mitbih_windows.npz"
INDEX_CSV     = BASE_DIR / "web" / "test_samples" / "index.csv"
OUT_DIR       = BASE_DIR / "ui_test_files"

def ensure_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1).astype(np.float32)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- check files exist ---
    if not PROCESSED_NPZ.exists():
        raise FileNotFoundError(f"Processed NPZ not found: {PROCESSED_NPZ}")
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Index CSV not found: {INDEX_CSV}")

    d = np.load(PROCESSED_NPZ, allow_pickle=True)

    # Your log confirmed these keys:
    X_key = "X"
    y_key = "y_class"
    hr_key = "y_hr"

    X = d[X_key]  # (N,1,L)
    X1 = X[:, 0, :]  # (N,L)
    N, L = X1.shape

    y = d[y_key].astype(int).reshape(-1) if y_key in d.files else None
    hr = d[hr_key].astype(float).reshape(-1) if hr_key in d.files else None

    idx_df = pd.read_csv(INDEX_CSV)

    meta_rows = []
    for _, r in idx_df.iterrows():
        src_i = int(r["source_index"])
        file_base = str(r["file_base"])

        sig = ensure_1d(X1[src_i])

        npy_path = OUT_DIR / f"{file_base}.npy"
        csv_path = OUT_DIR / f"{file_base}.csv"

        np.save(npy_path, sig)
        np.savetxt(csv_path, sig, delimiter=",")

        meta_rows.append({
            "file_base": file_base,
            "saved_npy": str(npy_path),
            "saved_csv": str(csv_path),
            "source_index": src_i,
            "label_from_index_csv": int(r["label"]) if "label" in idx_df.columns else None,
            "label_from_npz": (int(y[src_i]) if y is not None else None),
            "hr_from_index_csv": (float(r["hr_bpm"]) if "hr_bpm" in idx_df.columns else None),
            "hr_from_npz": (float(hr[src_i]) if hr is not None else None),
            "length": int(len(sig)),
        })

    out_index = OUT_DIR / "generated_index.csv"
    pd.DataFrame(meta_rows).to_csv(out_index, index=False)

    print("✅ Done.")
    print(f"✅ Read NPZ: {PROCESSED_NPZ}")
    print(f"✅ Read index: {INDEX_CSV}")
    print(f"✅ Saved files in: {OUT_DIR}")
    print(f"✅ Summary index: {out_index}")

if __name__ == "__main__":
    main()
