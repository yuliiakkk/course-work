import os
import cv2
import torch
import numpy as np
from PIL import Image
from model import UNet
from lama.saicinpainting.training.trainers import load_checkpoint
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.evaluation.data import pad_img_to_modulo
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from omegaconf import OmegaConf

# 📁 Шляхи
IMAGE_DIR = "dataset/images"
RESULTS_DIR = "results"
MODEL_PATH = "model/final_model.pth"
LAMA_MODEL_DIR = "lama/big-lama"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Завантаження UNet
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Не знайдено model/final_model.pth — спочатку натренуйте модель через train.py")

unet = UNet(in_channels=4, out_channels=1).to(DEVICE)
unet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
unet.eval()

# Завантаження CLIPSeg
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

# Завантаження LaMa
lama_cfg = OmegaConf.load(os.path.join(LAMA_MODEL_DIR, "config.yaml"))
lama_model = load_checkpoint(path=os.path.join(LAMA_MODEL_DIR, "models", "best.ckpt"),
                             strict=False, map_location="cpu")
lama_model.freeze()
lama_model.to(DEVICE)

# 🔤 Ввід об'єкта
prompt = input("Введи назву об'єкта для видалення: ").strip()
os.makedirs(RESULTS_DIR, exist_ok=True)

# 📂 Обхід усіх зображень
for filename in sorted(os.listdir(IMAGE_DIR)):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    print(f"🔍 Обробка: {filename}")

    image_pil = Image.open(image_path).convert("RGB")
    image_cv = cv2.imread(image_path)

    # CLIPSeg
    inputs = clipseg_processor(text=prompt, images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = clipseg_model(**inputs)
    clipseg_mask = outputs.logits[0][0].cpu().numpy()
    clipseg_mask = (clipseg_mask > 0.3).astype(np.uint8) * 255

    # Resize CLIPSeg маски
    clipseg_mask_resized = cv2.resize(clipseg_mask, image_pil.size, interpolation=cv2.INTER_NEAREST)

    # Об'єднання зображення + маски → UNet
    image_tensor = torch.from_numpy(image_cv[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    clipseg_tensor = torch.from_numpy(clipseg_mask_resized).unsqueeze(0).float() / 255.0
    input_tensor = torch.cat([image_tensor, clipseg_tensor], dim=0).unsqueeze(0).to(DEVICE)

    # UNet
    with torch.no_grad():
        predicted_mask = unet(input_tensor)[0][0].cpu().numpy()
    predicted_mask_bin = (predicted_mask > 0.5).astype(np.uint8) * 255

    # LaMa
    batch = {
        "image": cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB).astype("uint8"),
        "mask": predicted_mask_bin.astype("uint8")
    }
    batch = pad_img_to_modulo(batch, 8)
    batch = move_to_device(batch, DEVICE)

    with torch.no_grad():
        result = lama_model(batch)[0].permute(1, 2, 0).cpu().numpy().astype("uint8")

    # 💾 Збереження
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result_path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(result_path, result_bgr)
    print(f"✅ Збережено результат: {result_path}")