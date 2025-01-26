import torch
from torch import load
from torch.utils.data import DataLoader
from model import model, device
from dataset import test_data

model.load_state_dict(load("models/mnist_model.pth", weights_only=True))
model.eval()

test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nAccuracy: {accuracy:.2f}%")
