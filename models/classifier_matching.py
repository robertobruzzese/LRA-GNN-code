import torch.nn as nn

class AgeGroupClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  # net.0
            nn.ReLU(),                 # net.1
            nn.Linear(64, 64),         # net.2
            nn.ReLU(),                 # net.3
            nn.Linear(64, num_classes) # net.4
        )

    def forward(self, x):
        return self.net(x)
    
