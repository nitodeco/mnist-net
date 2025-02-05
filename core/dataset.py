from torchvision import datasets
import torchvision.transforms as transforms

base_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.1)),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

training_data = datasets.MNIST(
    root="data",
    train=True,
    transform=train_transform,
    download=True,
)

test_data = datasets.MNIST(root="data", train=False, transform=base_transform)
