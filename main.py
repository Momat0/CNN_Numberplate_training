import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_DIR = "classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "val")
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
IMG_SIZE = (32, 32)
MODEL_PATH = "char_classifier.pth"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------- TRANSFORMS ----------
train_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
valid_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------- DATA ----------
train_ds = ImageFolder(TRAIN_DIR, transform=train_tf)
valid_ds = ImageFolder(VALID_DIR, transform=valid_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)
NUM_CLASSES = len(train_ds.classes)
CLASS_NAMES = train_ds.classes  # 🔑 Save class names

# ---------- MODEL ----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------- TRACKING ----------
train_losses, valid_losses = [], []
train_accs, valid_accs = [], []

# ---------- TRAIN ----------
for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    valid_loss = val_loss / len(valid_loader)
    valid_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"T Loss: {train_loss:.4f} T Acc: {train_acc:.4f} | "
          f"V Loss: {valid_loss:.4f} V Acc: {valid_acc:.4f}")

    # --- Save stats ---
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)

# ---------- SAVE MODEL ----------
torch.save({
    "model_state": model.state_dict(),
    "class_names": CLASS_NAMES
}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ---------- PLOT METRICS ----------
epochs = range(1, EPOCHS+1)

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, valid_losses, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "loss_plot.png"))
plt.close()

plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, valid_accs, label="Valid Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "acc_plot.png"))
plt.close()

print(f"Saved plots to {PLOT_DIR}/loss_plot.png and {PLOT_DIR}/acc_plot.png")
