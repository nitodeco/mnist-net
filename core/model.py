import torch
from torch import nn

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class Network(nn.Module):
    def __init__(self, layer_sizes=[256, 128], dropout_rates=[0.3, 0.3]):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        prev_size = 28 * 28

        for size, dropout in zip(layer_sizes, dropout_rates):
            layers.extend([nn.Linear(prev_size, size), nn.ReLU(), nn.Dropout(dropout)])
            prev_size = size

        layers.append(nn.Linear(prev_size, 10))

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, batch):
        image = batch[0] if isinstance(batch, tuple) else batch
        image = self.flatten(image)
        image = image.view(-1, 28 * 28)
        logits = self.linear_relu_stack(image)
        return logits


model = Network().to(device)
