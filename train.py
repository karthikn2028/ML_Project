import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
# === 1. Dataset Paths ===
# === 1. Dataset Paths ===
train_dir = r"C:\Users\Karthikeyan\Downloads\ML_Project\split folder\train"
val_dir = r"C:\Users\Karthikeyan\Downloads\ML_Project\split folder\val"
test_dir = r"C:\Users\Karthikeyan\Downloads\ML_Project\split folder\test"

# === 2. Image Preprocessing ===
IMG_SIZE = 224  # ConvNeXt expects 224x224

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === 3. Load Dataset ===
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=val_test_transforms)

# Automatically save class names for Streamlit app
# (Indha file app.py irukara folder-laye save a 
with open("class_names.json", "w") as f:
    json.dump(train_data.classes, f, indent=4)
print("💾 Saved class_names.json automatically!")

# === 4. Data Loaders ===
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

print(f"✅ Loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
print(f"📂 Total Classes: {len(train_data.classes)}")

# === 5. Model Setup (ConvNeXt Tiny) ===
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

# Freeze feature extractor layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head for our number of bird classes
num_classes = len(train_data.classes)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

model = model.to(device)

# === 6. Hyperparameters ===
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.classifier[2].parameters(), lr=learning_rate, weight_decay=1e-4)
num_epochs = 10

# === 7. Training Loop ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation Phase
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

    val_acc = correct.double() / len(val_data)
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Val Acc: {val_acc:.4f}")

print("✅ Training complete!")

# === 8. Testing Phase ===
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

test_accuracy = 100.0 * correct / total
print(f"🎯 Test Accuracy: {test_accuracy:.2f}%")

# === 9. Save Model Weights ===
torch.save(model.state_dict(), "convnext_bird_model.pth")
print("💾 Model saved successfully as convnext_bird_model.pth")