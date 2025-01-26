from torch import nn, save
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from .dataset import training_data
from .model import model, device
import os

epochs = 10
learning_rate = 0.001
batch_size = 32
weight_decay = 0.01


def train():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.1)),
        ]
    )

    train_data_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
    )

    gradient = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        gradient, mode="min", factor=0.1, patience=2
    )

    loss_fn = nn.CrossEntropyLoss()
    steps = len(train_data_loader)
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0
        for x, (images, labels) in enumerate(train_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_fn(output, labels)

            gradient.zero_grad()
            loss.backward()
            gradient.step()

            epoch_loss += loss.item()

            if (x + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Step {x+1}/{steps}, Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_data_loader)
        scheduler.step(avg_loss)

        current_lr = gradient.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save(model.state_dict(), "models/mnist_model_best.pth")

    os.makedirs("models", exist_ok=True)
    save(model.state_dict(), "models/mnist_model.pth")
    print("Training complete. Best model saved to models/mnist_model_best.pth")
