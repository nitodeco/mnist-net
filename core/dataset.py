from torchvision import datasets
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomRotation(8),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ]
)

training_data = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True,
)
test_data = datasets.MNIST(root="data", train=False, transform=transform)
