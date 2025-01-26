import torch
from torch import nn

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, batch):
        image = batch[0] if isinstance(batch, tuple) else batch
        image = self.flatten(image)
        image = image.view(-1, 28 * 28)
        logits = self.linear_relu_stack(image)
        return logits


model = Network().to(device)
