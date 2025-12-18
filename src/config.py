from dataclasses import dataclass

@dataclass
class Config:
    # --------- Dataset ----------
    dataset_root: str = "data/raw_mitbih"
    processed_path: str = "data/processed/mitbih_windows.npz"

    # If True, will download records via wfdb from PhysioNet (needs internet when you run)
    download_if_missing: bool = True

    # Which records to use (MIT-BIH common set)
    records: tuple = (
        "100","101","102","103","104","105","106","107","108","109",
        "111","112","113","114","115","116","117","118","119",
        "121","122","123","124","200","201","202","203","205","207",
        "208","209","210","212","213","214","215","217","219",
        "220","221","222","223","228","230","231","232","233","234"
    )

    # --------- Signal Processing ----------
    fs_target: int = 360          # MIT-BIH sampling rate
    window_seconds: float = 12.0   # window length for model input
    step_seconds: float = 6.0     # stride (overlap)
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0

    # Minimum beats needed in a window to compute HR label
    min_beats_for_hr: int = 4

    # --------- Labels ----------
    num_classes: int = 5   # AAMI 5-class: N,S,V,F,Q

    # --------- Training ----------
    seed: int = 42
    batch_size: int = 128
    epochs: int = 20        # run full epochs, no early stopping
    lr: float = 8e-4
    weight_decay: float = 1e-4
    lambda_hr: float = 0.2    # slightly stronger HR learning
    
    # Stability
    dropout: float = 0.2
    grad_clip: float = 1.0    # prevents exploding gradients
    label_smoothing: float = 0.05
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    
    # System
    use_cuda: bool = True
    num_workers: int = 0


    # Regularization / early stopping
    dropout: float = 0.2
    patience: int = 5

    # --------- System ----------
    use_cuda: bool = True
    num_workers: int = 2

use_focal_loss: bool = False      # ✅ OFF
label_smoothing: float = 0.05     # ✅ ON (helps stability)
lambda_hr: float = 0.25           # keeps HR good, less effect on class task

