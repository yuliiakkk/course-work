import os
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from model import UNet
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.trainers import load_checkpoint
from lama.saicinpainting.evaluation.data import pad_img_to_modulo
from omegaconf import OmegaConf

PROMPT = input("Введи назву об'єкта для видалення: ").strip()
USE_UNET = False
IMG_SIZE = 512
LAMA_MODEL_PATH = "lama/big-lama"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/generated_masks"
RESULTS_DIR = "results"
MODEL_WEIGHTS = "model/final_model.pth"

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

lama_config = OmegaConf.load(os.path.join(LAMA_MODEL_PATH, 'config.yaml'))
lama_model = load_checkpoint(path=os.path.join(LAMA_MODEL_PATH, 'models', 'best.ckpt'), strict=False, map_location='cpu')
lama_model.freeze()
lama_model.to(DEVICE)

image_filenames = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])
for filename in image_filenames:
    print(f"\n📷 Обробка: {filename}")
    image_path = os.path.join(IMAGE_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename.replace(".jpg", ".png"))
    result_path = os.path.join(RESULTS_DIR, filename)

    image = Image.open(image_path).convert("RGB")
    inputs = clipseg_processor(text=PROMPT, images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = clipseg_model(**inputs)

    clipseg_mask = outputs.logits[0][0].cpu().numpy()
    clipseg_mask_bin = (clipseg_mask > 0.3).astype(np.uint8) * 255
    clipseg_mask_bin = cv2.resize(clipseg_mask_bin, image.size, interpolation=cv2.INTER_NEAREST)

    if USE_UNET:
        print("Уточнення через UNet...")
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(image)
        clipseg_tensor = torch.from_numpy(cv2.resize(clipseg_mask, (IMG_SIZE, IMG_SIZE))).float().unsqueeze(0)
        input_tensor = torch.cat([img_tensor, clipseg_tensor], dim=0).unsqueeze(0).to(DEVICE)

        model = UNet().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            pred_mask = model(input_tensor).squeeze().cpu().numpy()
            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
            used_mask = cv2.resize(pred_mask_bin, image.size, interpolation=cv2.INTER_NEAREST)
    else:
        used_mask = clipseg_mask_bin

    image_cv = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    batch = {
        "image": image_rgb.astype("uint8"),
        "mask": used_mask.astype("uint8")
    }
    batch = pad_img_to_modulo(batch, 8)
    batch = move_to_device(batch, DEVICE)

    with torch.no_grad():
        result = lama_model(batch)[0].permute(1, 2, 0).cpu().numpy()

    result_bgr = cv2.cvtColor(result.astype("uint8"), cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path, result_bgr)
    print(f"Готово → {result_path}")

print("\nОбробка завершена для всіх зображень.")
