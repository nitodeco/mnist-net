import torch
from torch import load
from torch.utils.data import DataLoader
from .model import Network, device
from .dataset import test_data


def evaluate(confidence_threshold=0.0):
    model = Network().to(device)

    checkpoint = load("models/mnist_model.pth", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            if confidence_threshold > 0:
                confident_predictions = confidence >= confidence_threshold
                total += confident_predictions.sum().item()
                correct += ((predicted == labels) & confident_predictions).sum().item()

                c = (predicted == labels).squeeze() & confident_predictions
                for i in range(len(labels)):
                    label = labels[i]
                    if confident_predictions[i]:
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
            else:
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"Accuracy of class {i}: {class_acc:.2f}%")
