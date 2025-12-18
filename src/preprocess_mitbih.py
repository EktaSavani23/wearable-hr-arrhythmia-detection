import os
import numpy as np
from scipy.signal import butter, filtfilt
import wfdb
from tqdm import tqdm

# --- AAMI 5-class mapping from beat symbols ---
# N: Normal + bundle branch etc.
AAMI_N = set(["N","L","R","e","j"])
# S: supraventricular ectopic
AAMI_S = set(["A","a","J","S"])
# V: ventricular ectopic
AAMI_V = set(["V","E"])
# F: fusion
AAMI_F = set(["F"])
# Q: paced/unknown
AAMI_Q = set(["/","f","Q","?","|"])

def beat_to_aami(symbol: str) -> int:
    if symbol in AAMI_N: return 0
    if symbol in AAMI_S: return 1
    if symbol in AAMI_V: return 2
    if symbol in AAMI_F: return 3
    return 4  # Q

def bandpass_filter(x, fs, low, high, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def zscore(x, eps=1e-8):
    return (x - x.mean()) / (x.std() + eps)

def ensure_record_available(record: str, root: str, download: bool):
    """Ensure WFDB record exists locally; optionally download from PhysioNet when you run."""
    dat_path = os.path.join(root, record + ".dat")
    hea_path = os.path.join(root, record + ".hea")
    atr_path = os.path.join(root, record + ".atr")

    if os.path.exists(dat_path) and os.path.exists(hea_path) and os.path.exists(atr_path):
        return

    if not download:
        raise FileNotFoundError(f"Record {record} not found in {root}. Download it or set download_if_missing=True.")

    # Download from PhysioNet
    wfdb.dl_database("mitdb", dl_dir=root, records=[record])

def build_windows(cfg):
    fs = cfg.fs_target
    win_len = int(cfg.window_seconds * fs)
    step = int(cfg.step_seconds * fs)

    X_list, y_class_list, y_hr_list = [], [], []

    for rec in tqdm(cfg.records, desc="Preprocessing MIT-BIH"):
        ensure_record_available(rec, cfg.dataset_root, cfg.download_if_missing)

        # Load ECG signal (use channel 0 by default)
        record = wfdb.rdrecord(os.path.join(cfg.dataset_root, rec))
        sig = record.p_signal[:, 0].astype(np.float32)

        # Load beat annotations (R-peak locations + beat symbols)
        ann = wfdb.rdann(os.path.join(cfg.dataset_root, rec), "atr")
        r_peaks = np.array(ann.sample, dtype=np.int64)
        symbols = np.array(ann.symbol)

        # Filter + normalize
        sig_f = bandpass_filter(sig, fs, cfg.bandpass_low, cfg.bandpass_high)
        sig_f = zscore(sig_f)

        # Slide windows
        for start in range(0, len(sig_f) - win_len, step):
            end = start + win_len
            x_win = sig_f[start:end].copy()

            # Find beats inside this window
            mask = (r_peaks >= start) & (r_peaks < end)
            peaks_in = r_peaks[mask]
            sym_in = symbols[mask]

            # Need at least one beat to label class
            if len(sym_in) < 1:
                continue

            # Window class label = majority AAMI class of beats in window
            classes = [beat_to_aami(s) for s in sym_in]
            y_class = int(np.bincount(classes, minlength=cfg.num_classes).argmax())

            # HR label from RR intervals within window
            # HR label from RR intervals within window (robust trimmed mean)
            if len(peaks_in) < cfg.min_beats_for_hr:
                continue
            
            rr = np.diff(peaks_in) / fs
            
            # Keep realistic RR intervals only (30–240 bpm)
            rr = rr[(rr > 0.25) & (rr < 2.0)]
            if len(rr) < (cfg.min_beats_for_hr - 1):
                continue
            
            rr_sorted = np.sort(rr)
            
            # Trim 10% from each side (robust against outliers)
            k = max(1, int(0.1 * len(rr_sorted)))
            if len(rr_sorted) > 2 * k:
                rr_trim = rr_sorted[k:len(rr_sorted) - k]
            else:
                rr_trim = rr_sorted
            
            rr_mean = float(np.mean(rr_trim))
            if rr_mean <= 0:
                continue
            
            y_hr = 60.0 / rr_mean
            y_hr = float(np.clip(y_hr, 30.0, 220.0))


            X_list.append(x_win[None, :])  # (1, L)
            y_class_list.append(y_class)
            y_hr_list.append(y_hr)

    X = np.stack(X_list, axis=0).astype(np.float32)          # (N,1,L)
    y_class = np.array(y_class_list, dtype=np.int64)         # (N,)
    y_hr = np.array(y_hr_list, dtype=np.float32)             # (N,)

    return X, y_class, y_hr

def save_processed(cfg):
    X, y_class, y_hr = build_windows(cfg)
    np.savez(cfg.processed_path, X=X, y_class=y_class, y_hr=y_hr)
    print(f"✅ Saved: {cfg.processed_path}")
    print("Shapes:", X.shape, y_class.shape, y_hr.shape)
