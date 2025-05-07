import os
import torch
from PIL import Image
import numpy as np
import cv2
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

IMAGE_DIR = "dataset/images"
TEXT_DIR = "dataset/landmarks"
OUTPUT_DIR = "dataset/generated_masks"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Завантаження моделі CLIPSeg...")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

image_filenames = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nГенеруємо маски для {len(image_filenames)} зображень...")
for filename in image_filenames:
    image_path = os.path.join(IMAGE_DIR, filename)
    text_path = os.path.join(TEXT_DIR, filename.replace(".jpg", ".txt"))
    output_path = os.path.join(OUTPUT_DIR, filename.replace(".jpg", ".png"))

    if not os.path.exists(text_path):
        print(f"Пропущено (немає опису): {filename}")
        continue

    image = Image.open(image_path).convert("RGB")
    with open(text_path, 'r') as f:
        prompt = f.read().strip()

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    mask = outputs.logits[0][0].cpu().numpy()
    mask = (mask > 0.3).astype(np.uint8) * 255

    mask_resized = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(output_path, mask_resized)
    print(f"Маска збережена: {output_path}")

print("\nГенерацію завершено.")
