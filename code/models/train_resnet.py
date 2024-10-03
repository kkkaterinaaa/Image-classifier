import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../datasets'))
from load_data import load_cifar10


if not os.path.exists('../../models'):
    os.makedirs('../../models')

train_loader, test_loader = load_cifar10(batch_size=32, data_dir='../../data')

model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(model, train_loader, criterion, optimizer, epochs=2):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")

        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")

        print(f"Epoch {epoch + 1}, Average Loss: {running_loss / len(train_loader):.4f}")

    print("Finished Training")


train_model(model, train_loader, criterion, optimizer, epochs=2)

model_save_path = '../../models/resnet_cifar10.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model: {accuracy:.2f}%")


test_model(model, test_loader)