import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from IPython import embed
from PIL import Image
from skimage import color
from torchvision import transforms
from tqdm import tqdm

from models.resnet import ResNetColorizationModel
from models.vgg import VGGColorizationModel
from models.vit import ViTColorizationModel
from utils.colorization_utils import ColorizationUtils
from utils.predicting_utils import PredictingUtils


def generate_random_ab(hw):
    """Generate random **ab** channels in [-127, 128] with shape (2, H, W)."""
    return np.random.uniform(-128, 127, size=(2, *hw)).astype(np.float32)


if __name__ == "__main__":
    with open("src/configs/vgg_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model_ckpt_path = f"{config['output']['best_model_dir']}/best_model.pth"
    if config["model"]["name"] == "resnet":
        if os.path.exists(model_ckpt_path):
            model = ResNetColorizationModel(pretrained=False)
            state = torch.load(model_ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            print("Model loaded successfully.")
    elif config["model"]["name"] == "vit":
        if os.path.exists(model_ckpt_path):
            model = ViTColorizationModel(pretrained=False)
            state = torch.load(model_ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            print("Model loaded successfully.")
    elif config["model"]["name"] == "vgg":
        if os.path.exists(model_ckpt_path):
            model = VGGColorizationModel(pretrained=False)
            state = torch.load(model_ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            print("Model loaded successfully.")

    img_paths = PredictingUtils.collect_images(
        config["testing"]["test_dir"], config["testing"]["subset_percent"]
    )
    if not config["testing"]["visualisation"]:
        out_folder = config["testing"]["predicted_image_dir"]
        os.makedirs(out_folder, exist_ok=True)
    else:
        out_folder = config["testing"]["visualisation_dir"]
        os.makedirs(out_folder, exist_ok=True)

    target_size = tuple(config["data"]["image_size"])
    resize_transform = transforms.Resize(target_size)

    for img in tqdm(img_paths, desc="Predicting"):
        pil_in = np.array(Image.open(img).convert("RGB"))
        out_img_eccv16 = PredictingUtils.predict(model, device, pil_in)

        plt.imshow(out_img_eccv16)
        plt.show()
