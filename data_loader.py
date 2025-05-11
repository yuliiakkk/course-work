import os
import torch
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

#Конфігурація
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512

#CLIPSeg модель
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

#Уніфіковані трансформації (з CenterCrop для кращої якості)
def get_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

#Torch dataset клас для навчання
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

#Генерація CLIPSeg-масок у пакетному режимі
def generate_all_masks(image_dir, text_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

    print(f"Генеруємо маски для {len(image_filenames)} зображень...")
    for filename in image_filenames:
        image_path = os.path.join(image_dir, filename)
        text_path = os.path.join(text_dir, filename.replace(".jpg", ".txt"))
        output_path = os.path.join(output_dir, filename.replace(".jpg", ".png"))

        if not os.path.exists(text_path):
            print(f"Пропущено (немає опису): {filename}")
            continue

        image = Image.open(image_path).convert("RGB")
        with open(text_path, 'r') as f:
            prompt = f.read().strip()

        inputs = clipseg_processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = clipseg_model(**inputs)
        mask = outputs.logits[0][0].cpu().numpy()
        mask = (mask > 0.3).astype(np.uint8) * 255

        mask_resized = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_path, mask_resized)
        print(f"Маска збережена: {output_path}")