import argparse
import os
import sys
import time
import warnings

import numpy as np
import psutil  # Import psutil
import torch
import torch.nn.functional as F
import yaml
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from models.resnet import ResNetColorizationModel
from models.vgg import VGGColorizationModel
from models.vit import ViTColorizationModel
from utils.colorization_utils import ColorizationUtils
from utils.predicting_utils import PredictingUtils


def get_memory_usage_mb():
    """Returns the memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with a colorization model.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/vgg_config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--use_quantized",
        action="store_true",
        help="Load and use the INT8 dynamic quantized version of the model.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Memory & Model Loading ---
    mem_before_load = get_memory_usage_mb()
    model_type = "INT8 Quantized" if args.use_quantized else "Float32"

    if args.use_quantized:
        print("Loading INT8 Dynamic Quantized Model...")
        model_ckpt_path = (
            f"{config['output']['best_model_dir']}/best_model_dynamic_int8.pth"
        )
        model = torch.load(model_ckpt_path, map_location=device, weights_only=False)
    else:
        print("Loading Float32 Model...")
        model = VGGColorizationModel(pretrained=False)
        model_ckpt_path = f"{config['output']['best_model_dir']}/best_model.pth"
        model.load_state_dict(
            torch.load(model_ckpt_path, map_location=device, weights_only=False)
        )

    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_ckpt_path}")

    mem_after_load = get_memory_usage_mb()
    model_mem_footprint = mem_after_load - mem_before_load

    # --- Prediction, Timing, & RAM Tracking ---
    img_paths = PredictingUtils.collect_images(
        config["testing"]["test_dir"], config["testing"]["subset_percent"]
    )
    target_size = tuple(config["data"]["image_size"])
    num_images = len(img_paths)
    model_uses_classification = config["model"].get("quantized_output", False)

    print(f"\nStarting prediction for {num_images} images with {model_type} model...")
    peak_mem_during_inference = mem_after_load
    start_time = time.time()

    for img_path in tqdm(img_paths, desc="Predicting"):
        pil_in = Image.open(img_path).convert("RGB")

        if model_uses_classification:
            ab_bins = ColorizationUtils.get_ab_bins().to(device)
            annealed_mean_temp = config["testing"]["annealed_mean_temp"]
            l_orig_tensor, l_resized_tensor = PredictingUtils.preprocess_img(
                np.array(pil_in), target_hw=target_size
            )
            l_resized_tensor = l_resized_tensor.to(device)
            with torch.no_grad():
                ab_logits = model(l_resized_tensor)
                ab_predicted = ColorizationUtils.annealed_mean_prediction(
                    ab_logits, ab_bins, temperature=annealed_mean_temp
                ).cpu()
            out_img = PredictingUtils.postprocess_tens(l_orig_tensor, ab_predicted)
        else:
            out_img = PredictingUtils.predict(model, device, np.array(pil_in))
            # plt.imshow(out_img)
            # plt.title("Colorized Image")
            # plt.axis("off")
            # plt.show()

        # Update peak memory usage
        current_mem = get_memory_usage_mb()
        if current_mem > peak_mem_during_inference:
            peak_mem_during_inference = current_mem

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / num_images if num_images > 0 else 0

    # --- Report Results ---
    print("\n--- 📊 Performance Results ---")
    print(f"Model Type: {model_type}")
    print(f"Total images processed: {num_images}")
    print("-" * 20)
    print("RAM Usage:")
    print(f"  - Model footprint: {model_mem_footprint:.2f} MB")
    print(f"  - Peak RAM during prediction: {peak_mem_during_inference:.2f} MB")
    print("-" * 20)
    print("Timing:")
    print(f"  - Total prediction time: {total_time:.4f} seconds")
    print(f"  - Average time per image: {avg_time_per_image:.4f} seconds")
