import torch
import yaml
import os
from tqdm import tqdm
from PIL import Image

from models.resnet import ResNetColorizationModel
from utils.predicting_utils import PredictingUtils
from utils.colorization_utils import ColorizationUtils


if __name__ == "__main__":
    with open("src/configs/resnet_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model_ckpt_path = f"{config["output"]["best_model_dir"]}/best_model.pth"
    if config["model"]["name"] == "resnet":
        if os.path.exists(model_ckpt_path):
            model = ResNetColorizationModel(pretrained=False)
            state = torch.load(model_ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            print("Model loaded successfully.")
        else:
            print("Model file not found. Please check the path.")
            raise FileNotFoundError(model_ckpt_path)

    img_paths = PredictingUtils.collect_images(config["testing"]["test_dir"], config["testing"]["subset_percent"])

    out_folder = config["testing"]["predicted_image_dir"]
    os.makedirs(out_folder, exist_ok=True)

    target_size = tuple(config["data"]["image_size"])
    for img in (tqdm(img_paths, desc="Predicting")):
        pil_in = Image.open(img).convert("RGB")
        lll, _ = ColorizationUtils.preprocess_image(pil_in, target_size)
        predicted_ab = PredictingUtils.predict(model, device, lll)

        lll_norm = lll.cpu() * 100.0
        ab_denorm = predicted_ab.cpu() * 128.0

        rgb_np_float = ColorizationUtils.reconstruct_image(lll_norm, ab_denorm)

        coloured_image = Image.fromarray((rgb_np_float * 255.0).clip(0, 255).astype("uint8"))

        out_path = f"{out_folder}/{os.path.basename(img)}"
        coloured_image.save(out_path)

    print(f"Finished. Results are in {out_folder}")
