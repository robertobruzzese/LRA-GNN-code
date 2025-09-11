import torch.nn as nn
import torch.nn.functional as F

class AgeGroupClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=10):
        super().__init__()
# 👇 Adattala per essere compatibile con il checkpoint
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 10)  # 10 gruppi di età (0-9)
        )

    def forward(self, x):
        return self.net(x)
