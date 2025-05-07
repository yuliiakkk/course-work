import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from model import UNet
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

DATASET_DIR = "dataset"
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

class ImageMaskDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.text_dir = os.path.join(root, "landmarks")
        self.filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename.replace(".jpg", ".png"))
        text_path = os.path.join(self.text_dir, filename.replace(".jpg", ".txt"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        with open(text_path, 'r') as f:
            prompt = f.read().strip()

        inputs = clipseg_processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = clipseg_model(**inputs)
        clipseg_mask = outputs.logits[0][0].cpu().numpy()
        clipseg_mask = (clipseg_mask > 0.3).astype(np.uint8) * 255

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            clipseg_mask = transforms.ToTensor()(Image.fromarray(clipseg_mask)).float()

        input_tensor = torch.cat([image, clipseg_mask], dim=0)

        return input_tensor, mask

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_dataset = ImageMaskDataset(DATASET_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet(in_channels=4, out_channels=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

print("\ Починаємо тренування...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch
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
print("\n✅ Модель збережено у model/final_model.pth")
