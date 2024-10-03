import torchvision
import torchvision.transforms as transforms
import torch


def load_cifar10(batch_size=32, data_dir='../../data'):
    """
    Load CIFAR-10 dataset with transformations for training and testing.
    Args:
        batch_size (int): Batch size for the DataLoader
        data_dir (str): Directory where the data will be stored

    Returns:
        train_loader, test_loader: PyTorch DataLoaders for CIFAR-10
    """

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 training and testing datasets
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # Create DataLoader for training and testing datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
