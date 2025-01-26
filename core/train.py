from torch import nn, save
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.transforms as transforms
from dataset import training_data
from model import model, device
import os

epochs = 2
learning_rate = 0.002

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.7,), (0.7,))]
)

train_data_loader = DataLoader(training_data, batch_size=10, shuffle=True)

gradient = Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

steps = len(train_data_loader)

for epoch in range(epochs):
    for x, (images, labels) in enumerate(train_data_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = loss_fn(output, labels)

        gradient.zero_grad()
        loss.backward()
        gradient.step()

        if (x + 1) % 100 == 0:
            print(
                f"Epochs [{epoch+1}/{epochs}], Step[{x+1}/{steps}], Loss: {loss.item():.4f}"
            )

os.makedirs("models", exist_ok=True)
save(model.state_dict(), "models/mnist_model.pth")
print("Training complete. Model saved to models/mnist_model.pth")
