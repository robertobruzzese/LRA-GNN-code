# ✅ Architettura ablation usata per UTKFACE nei tuoi train
import torch.nn as nn
class AgeGroupClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # net.0
            nn.ReLU(),                  # net.1
            nn.Dropout(0.5),            # net.2
            nn.Linear(128, 64),         # net.3
            nn.ReLU(),                  # net.4
            nn.Dropout(0.5),            # net.5
            nn.Linear(64, num_classes)  # net.6
        )
    def forward(self, x):
        return self.net(x)