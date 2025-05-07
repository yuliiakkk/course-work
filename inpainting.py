import os
import cv2
import torch
import numpy as np
from PIL import Image
from lama.saicinpainting.training.trainers import load_checkpoint
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.evaluation.data import pad_img_to_modulo
from omegaconf import OmegaConf

#Шляхи
IMAGE_PATH = "dataset/images/sample.jpg"
MASK_PATH = "dataset/generated_masks/temp_mask.png"
OUTPUT_PATH = "results/lama_output.jpg"
MODEL_PATH = "lama/big-lama"

#Пристрій
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

#Завантаження моделі
config_path = os.path.join(MODEL_PATH, 'config.yaml')
ckpt_path = os.path.join(MODEL_PATH, 'models', 'best.ckpt')
config = OmegaConf.load(config_path)
model = load_checkpoint(path=ckpt_path, strict=False, map_location='cpu')
model.freeze()
model.to(DEVICE)

image = cv2.imread(IMAGE_PATH)
mask = cv2.imread(MASK_PATH, 0)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

batch = {
    "image": image_rgb.astype("uint8"),
    "mask": mask.astype("uint8")
}
batch = pad_img_to_modulo(batch, 8)
batch = move_to_device(batch, DEVICE)

with torch.no_grad():
    result = model(batch)[0].permute(1, 2, 0).cpu().numpy()

result_bgr = cv2.cvtColor(result.astype("uint8"), cv2.COLOR_RGB2BGR)
cv2.imwrite(OUTPUT_PATH, result_bgr)
print(f"LaMa результат збережено в: {OUTPUT_PATH}")
