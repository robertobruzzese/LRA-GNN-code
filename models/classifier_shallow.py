import torch.nn as nn

class AgeGroupClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64,  num_classes)
        )

    def forward(self, x):
        return self.net(x)