# src/api_app.py
import io
import csv
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

from src.config import Config
from src.model import FastMultiTaskCNN


app = FastAPI(title="Wearable Arrhythmia + Heart Rate (Human-Readable API)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # coursework demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = Config()

BASE_DIR = Path(__file__).resolve().parents[1]  # .../code
CKPT_PATH = BASE_DIR / "results" / "checkpoints" / "best.pt"
HTML_PATH = BASE_DIR / "web" / "index.html"
PROC_PATH = BASE_DIR / "data" / "processed" / "mitbih_windows.npz"

# ---- IMPORTANT ----
# Your model returns 5 probabilities -> assume 5 classes: 0..4
# If your model is different, update this.
IDX_TO_LABEL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

# If you don't know exact medical names yet, keep these simple.
# You can later rename them (e.g., PAC/PVC/Fusion/Other).
LABEL_NAME = {
    0: "Normal Rhythm",
    1: "Arrhythmia (Class 1)",
    2: "Arrhythmia (Class 2)",
    3: "Arrhythmia (Class 3)",
    4: "Arrhythmia (Class 4)",
}

MODEL = None
DEVICE = torch.device("cpu")

WINDOW_LEN = None
HR_MEAN = None
HR_STD = None


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std()) + 1e-8
    return (x - mu) / sd


def _fix_length(x: np.ndarray, target_len: int) -> np.ndarray:
    x = x.astype(np.float32).reshape(-1)
    if len(x) > target_len:
        return x[-target_len:]
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)), mode="constant")
    return x


def _hr_status(hr_bpm: float) -> dict:
    # Simple, understandable categories for coursework demo
    if hr_bpm < 60:
        return {"category": "Low", "message": "Heart rate is below typical resting range (<60 bpm)."}
    if hr_bpm <= 100:
        return {"category": "Normal", "message": "Heart rate is in typical resting range (60–100 bpm)."}
    return {"category": "High", "message": "Heart rate is above typical resting range (>100 bpm)."}


def _rhythm_status(pred_label: int, confidence_percent: float) -> dict:
    # Binary decision: 0 = normal, else arrhythmia
    has_arr = (pred_label != 0)

    if not has_arr:
        msg = "No arrhythmia detected in this window (predicted Normal)."
    else:
        msg = (
            f"Arrhythmia suspected (predicted Label {pred_label}). "
            "Note: Heart rate can still be normal even if rhythm is abnormal."
        )

    return {
        "arrhythmia_detected": bool(has_arr),
        "message": msg,
        "confidence_percent": round(float(confidence_percent), 2),
    }


def _predict_from_signal(signal_1d: np.ndarray):
    if MODEL is None:
        return {"error": "Model not loaded. Check server logs."}
    if WINDOW_LEN is None:
        return {"error": "WINDOW_LEN not set. Ensure processed npz exists."}

    # 1) Prepare signal
    x = _fix_length(signal_1d, int(WINDOW_LEN))
    x = _zscore(x)
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,L)

    # 2) Model inference
    with torch.no_grad():
        logits, hr_hat, _ = MODEL(xt)

    probs_np = torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float64)
    pred_idx = int(np.argmax(probs_np))
    pred_label = int(IDX_TO_LABEL.get(pred_idx, pred_idx))

    # 3) HR denormalization (very important)
    hr_val = float(hr_hat.cpu().numpy().reshape(-1)[0])
    if HR_MEAN is not None and HR_STD is not None:
        hr_val = hr_val * float(HR_STD) + float(HR_MEAN)

    # Keep realistic values for demo
    hr_val = float(np.clip(hr_val, 30.0, 220.0))

    # 4) Build readable probabilities list
    pairs = []
    for i, p in enumerate(probs_np):
        lbl = int(IDX_TO_LABEL.get(i, i))
        pairs.append({
            "class_index": int(i),
            "label": lbl,
            "name": LABEL_NAME.get(lbl, f"Label {lbl}"),
            "prob": float(p),
            "percent": round(float(p) * 100.0, 2),
        })
    pairs.sort(key=lambda x: x["prob"], reverse=True)

    top1 = pairs[0]
    top3 = [{"label": x["label"], "name": x["name"], "percent": x["percent"]} for x in pairs[:3]]

    # 5) Human-friendly statuses
    rhythm = _rhythm_status(pred_label, top1["percent"])
    hr_info = _hr_status(hr_val)

    return {
        "summary": {
            "rhythm_prediction": top1["name"],
            "rhythm_label": int(pred_label),
            "confidence_percent": top1["percent"],
            "arrhythmia_detected": rhythm["arrhythmia_detected"],
            "heart_rate_bpm": round(hr_val, 1),
            "heart_rate_category": hr_info["category"],
        },
        "explanations": {
            "rhythm_message": rhythm["message"],
            "heart_rate_message": hr_info["message"],
            "note": "Rhythm detection (arrhythmia) and heart rate are different tasks. HR can be normal even if rhythm is abnormal.",
        },
        "top_3_classes": top3,
        "all_probabilities": pairs,   # keep for debugging
        "window_samples_used": int(WINDOW_LEN),
    }


class PredictJSON(BaseModel):
    signal: list[float]


@app.on_event("startup")
def startup_load():
    global MODEL, WINDOW_LEN, HR_MEAN, HR_STD

    print("=== Startup: Loading model + window length + HR stats ===")
    print("Base dir:", BASE_DIR)
    print("Checkpoint:", CKPT_PATH)
    print("Processed:", PROC_PATH)

    # Read window length + HR stats from processed file (best)
    if PROC_PATH.exists():
        d = np.load(PROC_PATH, allow_pickle=True)
        WINDOW_LEN = int(d["X"].shape[-1])
        print(f"✅ WINDOW_LEN={WINDOW_LEN}")

        if "y_hr" in d.files:
            y_hr = d["y_hr"].astype(np.float32).reshape(-1)
            HR_MEAN = float(y_hr.mean())
            HR_STD = float(y_hr.std() + 1e-8)
            print(f"✅ HR_MEAN={HR_MEAN:.3f} | HR_STD={HR_STD:.3f}")
        else:
            HR_MEAN = None
            HR_STD = None
            print("⚠️ y_hr not found in npz. HR may be raw model output.")
    else:
        WINDOW_LEN = 3600
        HR_MEAN = None
        HR_STD = None
        print("⚠️ Processed file not found. Using fallback WINDOW_LEN=3600")

    # Load model
    if not CKPT_PATH.exists():
        print(f"❌ Checkpoint not found: {CKPT_PATH}")
        MODEL = None
        return

    MODEL = FastMultiTaskCNN(num_classes=cfg.num_classes, dropout=cfg.dropout).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    MODEL.load_state_dict(ckpt["model_state"])
    MODEL.eval()
    print("✅ Model loaded and set to eval()")


@app.get("/health")
def health():
    # This endpoint is only for server check -> OK means API is running
    return {
        "status": "ok",
        "device": str(DEVICE),
        "ckpt_exists": CKPT_PATH.exists(),
        "processed_exists": PROC_PATH.exists(),
        "window_len": WINDOW_LEN,
        "hr_mean": HR_MEAN,
        "hr_std": HR_STD,
    }


@app.post("/predict_json")
def predict_json(payload: PredictJSON):
    sig = np.array(payload.signal, dtype=np.float32)
    if sig.size < 10:
        return {"error": f"Too few samples: {sig.size}. Provide a longer signal window."}
    return _predict_from_signal(sig)


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """
    Upload:
      - .csv / .txt : one column OR comma-separated floats
      - .npy        : numpy 1D array
    """
    try:
        content = await file.read()
        name = (file.filename or "").lower()

        if name.endswith(".npy"):
            sig = np.load(io.BytesIO(content)).astype(np.float32)

        elif name.endswith(".csv") or name.endswith(".txt"):
            text = content.decode("utf-8", errors="ignore")
            rows = list(csv.reader(io.StringIO(text)))
            vals = []
            for r in rows:
                for c in r:
                    c = c.strip()
                    if c:
                        vals.append(float(c))
            sig = np.array(vals, dtype=np.float32)

        else:
            return {"error": "Unsupported file. Upload .csv/.txt or .npy"}

        if sig.size < 10:
            return {"error": f"Too few samples in file: {sig.size}"}

        return _predict_from_signal(sig)

    except Exception as e:
        return {"error": f"Prediction failed: {type(e).__name__}: {e}"}


@app.get("/", response_class=HTMLResponse)
def home():
    if not HTML_PATH.exists():
        return HTMLResponse(f"<h3>index.html not found at {HTML_PATH}</h3>", status_code=404)
    return HTML_PATH.read_text(encoding="utf-8")
