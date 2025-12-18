import numpy as np
import torch
from torch.utils.data import Dataset

class WearableMultiTaskDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=False)
        self.X = torch.tensor(data["X"], dtype=torch.float32)            # (N,1,L)
        self.y_class = torch.tensor(data["y_class"], dtype=torch.long)   # (N,)
        self.y_hr = torch.tensor(data["y_hr"], dtype=torch.float32).unsqueeze(1)  # (N,1)
        self.hr_mean = self.y_hr.mean()
        self.hr_std = self.y_hr.std() + 1e-8
        self.y_hr = (self.y_hr - self.hr_mean) / self.hr_std


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_hr[idx]
