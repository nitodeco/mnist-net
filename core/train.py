from torch import nn, save
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim import lr_scheduler
from .dataset import training_data
from .model import model, device
import os
import time
import json


def save_model_config(layer_sizes, dropout_rates, timestamp=None):
    os.makedirs("models", exist_ok=True)
    config = {"layer_sizes": layer_sizes, "dropout_rates": dropout_rates}

    if timestamp:
        config_path = f"models/config_{timestamp}.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

    with open("models/config.json", "w") as f:
        json.dump(config, f)


def train(
    epochs=8, learning_rate=0.0015, batch_size=32, weight_decay=0.005, callback=None
):
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
        epoch_start_time = time.time()
        epoch_loss = 0
        losses = []

        for x, (images, labels) in enumerate(train_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_fn(output, labels)

            gradient.zero_grad()
            loss.backward()
            gradient.step()

            epoch_loss += loss.item()
            losses.append(loss.item())

            if callback:
                progress = (epoch * steps + x + 1) / (epochs * steps)
                time_per_step = (time.time() - epoch_start_time) / (x + 1)
                remaining_steps = (epochs * steps) - (epoch * steps + x + 1)
                eta = remaining_steps * time_per_step

                callback(progress, loss.item(), eta, epoch)

            if (x + 1) % batch_size == 0:
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
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            layer_sizes = [
                layer.out_features
                for layer in model.linear_relu_stack
                if isinstance(layer, nn.Linear)
            ][:-1]
            dropout_rates = [
                layer.p
                for layer in model.linear_relu_stack
                if isinstance(layer, nn.Dropout)
            ]
            save_model_config(layer_sizes, dropout_rates, timestamp)
            save(
                model.state_dict(),
                f"models/mnist_model_{timestamp}.pth",
            )

    os.makedirs("models", exist_ok=True)
    save(model.state_dict(), "models/mnist_model.pth")
    print("Training complete. Best model saved to models/mnist_model_best.pth")
