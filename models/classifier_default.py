import torch.nn as nn

class AgeGroupClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # net.0
            nn.ReLU(),                  # net.1
            nn.Dropout(0.5),            # net.2  (NO BatchNorm)
            nn.Linear(128, 64),         # net.3  ← il ckpt ha pesi qui
            nn.ReLU(),                  # net.4
            nn.Linear(64, num_classes)  # net.5
        )
    def forward(self, x):
        return self.net(x)