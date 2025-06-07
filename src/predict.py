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


def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if out_np.ndim == 2:
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def resize_img(img, HW=(256, 256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
    tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]

    return (tens_orig_l, tens_rs_l)


def postprocess_tens(tens_orig_l, out_ab, mode="bilinear"):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode="bilinear")
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))


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
        (tens_l_orig, tens_l_rs) = preprocess_img(pil_in, HW=(256, 256))
        tens_l_rs = tens_l_rs.to(device)
        img_bw = postprocess_tens(
            tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1)
        )
        out_img_eccv16 = postprocess_tens(tens_l_orig, model(tens_l_rs).cpu())

        plt.imshow(out_img_eccv16)
        plt.show()

        sys.exit()
        if not config["testing"]["visualisation"]:
            pil_in = Image.open(img).convert("RGB")
            lll, _ = ColorizationUtils.preprocess_image(pil_in, target_size)
            predicted_ab = PredictingUtils.predict(model, device, lll)

            rgb_np_float = ColorizationUtils.reconstruct_image(lll, predicted_ab)

            coloured_image = Image.fromarray(
                (rgb_np_float * 255.0).clip(0, 255).astype("uint8")
            )

            out_path = f"{out_folder}/{os.path.basename(img)}"
            coloured_image.save(out_path)
        else:
            pil_input = Image.open(img).convert("RGB")
            pil_resized = resize_transform(pil_input)

            AB_SCALE = ColorizationUtils.AB_SCALE  # 128.0
            L_SCALE = ColorizationUtils.L_SCALE  # 100.0

            # Keep ground-truth AB this time
            lll, gt_ab_norm = ColorizationUtils.preprocess_image(pil_input, target_size)

            predicted_ab_norm = PredictingUtils.predict(
                model, device, lll
            )  # (2,H,W), [-1,1]
            rgb_pred_np = ColorizationUtils.reconstruct_image(lll, predicted_ab_norm)
            predicted_image = Image.fromarray(
                (rgb_pred_np * 255.0).clip(0, 255).astype("uint8")
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Conversion from CIE-LAB.*negative Z values.*",
                )
                random_ab_unnorm = generate_random_ab(
                    (target_size[0], target_size[1])
                )  # un-normalised ±128
                rgb_rand_np = ColorizationUtils.reconstruct_image(
                    lll, torch.from_numpy(random_ab_unnorm / AB_SCALE)
                )
                random_image = Image.fromarray(
                    (rgb_rand_np * 255.0).clip(0, 255).astype("uint8")
                )

            gt_np = np.array(pil_resized)
            gray_np = np.array(pil_resized.convert("L"))

            gt_ab_unnorm = (gt_ab_norm.detach().cpu().numpy()) * AB_SCALE  # (2,H,W)
            predicted_ab_unnorm = (predicted_ab_norm.detach().cpu().numpy()) * AB_SCALE

            zero_ab_unnorm = np.zeros_like(predicted_ab_unnorm)  # (2,H,W)
            rgb_zero_np = ColorizationUtils.reconstruct_image(
                lll, torch.from_numpy(zero_ab_unnorm / AB_SCALE)
            )
            zero_ab_image = Image.fromarray(
                (rgb_zero_np * 255.0).clip(0, 255).astype("uint8")
            )

            mse_pred = np.mean((predicted_ab_unnorm - gt_ab_unnorm) ** 2)
            mse_zero = np.mean((zero_ab_unnorm - gt_ab_unnorm) ** 2)
            mse_rand = np.mean((random_ab_unnorm - gt_ab_unnorm) ** 2)

            # Plot
            fig, axes = plt.subplots(1, 5, figsize=(15, 4))

            axes[0].imshow(gray_np, cmap="gray")
            axes[0].set_title("Grayscale")

            axes[1].imshow(gt_np)
            axes[1].set_title("Ground truth")

            axes[2].imshow(predicted_image)
            axes[2].set_title(f"Predicted\nMSE={mse_pred:.1f}")

            axes[3].imshow(zero_ab_image)
            axes[3].set_title(f"ab = 0\nMSE={mse_zero:.1f}")

            axes[4].imshow(random_image)
            axes[4].set_title(f"Randomized\nMSE={mse_rand:.1f}")

            for ax in axes:
                ax.axis("off")
            fig.tight_layout()

            viz_path = os.path.join(out_folder, f"viz_{os.path.basename(img)}")
            fig.savefig(viz_path, dpi=150)
            plt.close(fig)
