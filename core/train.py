from torch import nn, save
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import lr_scheduler
from .dataset import training_data
from .model import model, device
import os
import time
import torch


def train(epochs=10, learning_rate=0.001, batch_size=32, callback=None):
    train_data_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    gradient = Adam(model.parameters(), lr=learning_rate)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        gradient, mode="min", factor=0.1, patience=2, verbose=True
    )

    loss_fn = nn.CrossEntropyLoss()
    steps = len(train_data_loader)

    print(f"\nStarting training with {epochs} epochs...")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0
        correct = 0
        total = 0

        for x, (images, labels) in enumerate(train_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            gradient.zero_grad()

            output = model(images)
            loss = loss_fn(output, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            gradient.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if callback:
                progress = (epoch * steps + x + 1) / (epochs * steps)
                time_per_step = (time.time() - epoch_start_time) / (x + 1)
                remaining_steps = (epochs * steps) - (epoch * steps + x + 1)
                eta = remaining_steps * time_per_step

                callback(progress, loss.item(), eta, epoch)

            if (x + 1) % (steps // 5) == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"Step {x+1}/{steps}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Accuracy: {100 * correct/total:.2f}%"
                )

        avg_loss = epoch_loss / len(train_data_loader)
        accuracy = 100 * correct / total
        epoch_time = time.time() - epoch_start_time

        print(
            f"\nEpoch {epoch+1}/{epochs} Summary:"
            f"\n - Average Loss: {avg_loss:.4f}"
            f"\n - Accuracy: {accuracy:.2f}%"
            f"\n - Time: {epoch_time:.2f}s"
        )

        scheduler.step(avg_loss)
        current_lr = gradient.param_groups[0]["lr"]
        print(f" - Learning Rate: {current_lr:.6f}\n")

    os.makedirs("models", exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save(model.state_dict(), f"models/mnist_model_{timestamp}.pth")
    print(f"\nTraining complete. Model saved with accuracy: {accuracy:.2f}%")
