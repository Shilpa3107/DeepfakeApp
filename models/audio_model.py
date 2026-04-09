import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ---------- CONFIG ----------
DATA_PATH = "data/audio_spectrograms"
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.0003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- TRANSFORMS ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- DATA ----------
train_data = ImageFolder(f"{DATA_PATH}/training", transform=transform)
val_data = ImageFolder(f"{DATA_PATH}/validation", transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

print("Classes:", train_data.class_to_idx)

# ---------- MODEL ----------
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze backbone (VERY IMPORTANT for stability)
for param in model.parameters():
    param.requires_grad = False

# Train only final layer
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ---------- LOSS ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# ---------- TRAIN ----------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ---------- VALIDATION ----------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            _, pred = torch.max(out, 1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {total_loss:.2f} | Val Acc: {acc:.2f}%")

# ---------- SAVE ----------
torch.save(model.state_dict(), "models/audio_model.pth")
print("✅ Audio model trained!")