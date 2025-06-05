import os

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from models.resnet import ResNetColorizationModel
from models.vit import ViTColorizationModel
from utils.colorization_utils import ColorizationUtils
from utils.predicting_utils import PredictingUtils

if __name__ == "__main__":
    with open("src/configs/vit_config.yaml", "r") as file:
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
            state = state["model_state_dict"]
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            print("Model loaded successfully.")

    img_paths = PredictingUtils.collect_images(
        config["testing"]["test_dir"], config["testing"]["subset_percent"]
    )

    out_folder = config["testing"]["predicted_image_dir"]
    os.makedirs(out_folder, exist_ok=True)

    target_size = tuple(config["data"]["image_size"])
    for img in tqdm(img_paths, desc="Predicting"):
        pil_in = Image.open(img).convert("RGB")
        lll, _ = ColorizationUtils.preprocess_image(pil_in, target_size)
        predicted_ab = PredictingUtils.predict(model, device, lll)

        rgb_np_float = ColorizationUtils.reconstruct_image(lll, predicted_ab)

        coloured_image = Image.fromarray(
            (rgb_np_float * 255.0).clip(0, 255).astype("uint8")
        )

        out_path = f"{out_folder}/{os.path.basename(img)}"
        coloured_image.save(out_path)

    print(f"Finished. Results are in {out_folder}")
