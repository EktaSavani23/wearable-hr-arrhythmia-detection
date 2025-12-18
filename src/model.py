import torch
import torch.nn as nn

class FastMultiTaskCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        # Global pooling → fixed-size vector (fast)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.shared = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.class_head = nn.Linear(128, num_classes)
        self.hr_head = nn.Linear(128, 1)

    def forward(self, x):
        z = self.encoder(x)                 # (B,128,L')
        z = self.pool(z).squeeze(-1)        # (B,128)
        z = self.shared(z)                  # (B,128)
        logits = self.class_head(z)         # (B,K)
        hr_hat = self.hr_head(z)            # (B,1)
        attn = None                         # no attention in fast model
        return logits, hr_hat, attn
