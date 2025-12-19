# Wearable HR & Arrhythmia Detection

This repository contains a quality-aware deep learning pipeline for joint **heart-rate (HR) estimation** and **arrhythmia classification** from wearable cardiac waveforms (ECG/PPG). The system is designed for realistic ambulatory conditions where signals are affected by **motion artefacts**, **baseline drift**, and **intermittent sensor contact**.

## What this project does
- Preprocesses wearable waveforms using detrending + band-pass filtering and robust normalisation
- Segments signals into fixed-length overlapping windows for learning and inference
- Uses a **Signal Quality Index (SQI)** to down-weight/ignore low-quality windows (improves robustness under noise)
- Trains a compact **1D CNN** with temporal aggregation
- Supports two outputs:
  - **Arrhythmia detection (classification)**: multi-class rhythm prediction
  - **Heart-rate estimation (regression)**: HR in bpm

## Typical repository structure
- `src/` : training / dataset / model / evaluation code
- `figures/` : report figures (PNG/PDF)
- `results/` : experiment outputs (ignored by git)
- `binary_test_samples/` : sample outputs
- `data/` : datasets and processed artifacts (ignored by git)

## Setup (Windows)
1) Create and activate virtual environment:
- `python -m venv .venv`
- `source .venv/Scripts/activate`

2) Install dependencies:
- `pip install -r requirements.txt`

## Training
Run from the project root:
- `python -m src.main_train`

## Running the API (FastAPI)
> If your API file is `src/api_app.py`:
- Start server:
  - `python -m uvicorn src.api_app:app --reload --host 0.0.0.0 --port 8000`
- Open Swagger UI:
  - http://127.0.0.1:8000/docs

> If your API file name is different, replace `src.api_app` with your module name (e.g., `src.api`).

## Evaluation outputs
Evaluation typically reports:
- **Classification:** Accuracy, Macro-F1, AUROC, Confusion Matrix
- **HR Estimation:** MAE (bpm), RMSE (bpm)

## Report figures
The report uses three figures stored in `figures/`:
- `fig_training_loss.png`
- `fig_training_accuracy.png`
- `fig_confusion_matrix.png`

## Example results (reported)
- Accuracy: 98.26%
- Macro-F1: 0.8845
- AUROC: 0.9952
- HR MAE: 4.87 bpm
- HR RMSE: 7.30 bpm

## Data note (GitHub size limit)
GitHub rejects files >100MB, so datasets and processed windows (e.g., `.npz`) are not committed. Use external storage (Drive/OneDrive) or Git LFS if needed.

## Author
Ekta Sanatkumar Savani  
University of Roehampton, London, UK  
Email: savanie@roehampton.ac.uk

## Repository
https://github.com/EktaSavani23/wearable-hr-arrhythmia-detection
