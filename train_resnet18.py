import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import json

# Loading json attribute
with open('config.json', 'r') as file:
    data = json.load(file)

# Load and Access to attributes
data_dir = data['data_dir']
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "validate")
test_dir = os.path.join(data_dir, "test")
num_epochs = data['num_epochs']
batch_size = data['batch']  # Use batch_size from config instead of hardcoded
saved_model_path = data['saved_model_path']
classes = data['classes']



# Improved data transforms with normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets and loaders
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

num_classes = len(train_dataset.classes)  # Get number of classes dynamically
print(f"Processing classes {num_classes}: ",classes)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
best_val_acc = 0.0
patience = 5  # Increased patience
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Training step
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # Track training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Calculate training accuracy
    train_acc = 100 * correct_train / total_train
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

    # Validation step
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Calculate validation accuracy
    val_acc = 100 * correct_val / total_val
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    # Check if validation accuracy improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        trigger_times = 0
        # Create directory if it doesn't exist
        os.makedirs(saved_model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(saved_model_path))
        print("✓ Best model updated.")
    else:
        trigger_times += 1
        print(f"⏳ No improvement. Early stop trigger: {trigger_times}/{patience}")
        if trigger_times >= patience:
            print("🛑 Early stopping.")
            break

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

# Test the model
model.eval()
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")