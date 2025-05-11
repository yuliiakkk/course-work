import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from data_loader import ImageMaskDataset, get_transform

DATASET_DIR = "dataset"
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = get_transform()
train_dataset = ImageMaskDataset(DATASET_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet(in_channels=4, out_channels=1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

print("\nПочинаємо тренування...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Епоха {epoch+1}/{EPOCHS} | Втрата: {total_loss/len(train_loader):.4f}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/final_model.pth")
print("\nМодель збережено у model/final_model.pth")