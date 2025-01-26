import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 100),
            nn.Linear(100, 50),
            nn.Linear(50, 10),
            nn.ReLU(),
        )

    def forward(self, image):
        image = self.flatten(image)
        logits = self.linear_relu_stack(image)
        return logits


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = Network().to(device)
