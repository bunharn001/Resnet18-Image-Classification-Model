import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os


# Paths to your data folders
data_dir = r"C:\Users\User\Resnet18-Image-Classification-Model"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")    # <-- Make sure you have a 'val' folder
test_dir = os.path.join(data_dir, "test")  # <-- Make sure you have a 'test' folder

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets and loaders
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: dog, cat
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
best_val_acc = 0.0  # Tracks the highest validation accuracy
patience = 3        # Number of epochs to wait before early stopping
trigger_times = 0   # Counter for non-improving epochs

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Training step
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()               # Clear previous gradients
        outputs = model(images)            # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                    # Backward pass
        optimizer.step()                   # Update weights
        running_loss += loss.item()

        # Track training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Calculate training accuracy
    train_acc = 100 * correct_train / total_train
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

    # Validation step
    model.eval()  # Set model to evaluation mode
    correct_val = 0
    total_val = 0
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Calculate validation accuracy
    val_acc = 100 * correct_val / total_val
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # Check if validation accuracy improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        trigger_times = 0  # Reset counter
        torch.save(model.state_dict(), os.path.join(data_dir, "best_model.pth"))  # Save best model
        print(":arrows_counterclockwise: Best model updated.")
    else:
        trigger_times += 1
        print(f":hourglass_flowing_sand: No improvement. Early stop trigger: {trigger_times}/{patience}")
        if trigger_times >= patience:
            print(":octagonal_sign: Early stopping.")
            break


# Save the trained model
output_model_path = os.path.join(data_dir, "TestTest.pth")
torch.save(model.state_dict(), output_model_path)
print(f"Model saved to {output_model_path}")

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")

