import argparse
import os

import torch
import torch.quantization
import yaml

from models.vgg import VGGColorizationModel


def print_model_size(
    model: torch.nn.Module, label: str = "", path: str = "temp_model.pth"
) -> None:
    """Helper function to print model size."""
    torch.save(model, path)
    size = os.path.getsize(path)
    print(f"Size of {label} model: {size / 1024 / 1024:.2f} MB")
    os.remove(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize a trained VGG model using the dynamic method."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/vgg_config.yaml",
        help="Path to the model's configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # --- 1. Load Trained Float Model ---
    # As requested, renamed this variable
    float_model = VGGColorizationModel(pretrained=False)

    model_ckpt_path = f"{config['output']['best_model_dir']}/best_model.pth"
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(
            f"Floating point model checkpoint not found at {model_ckpt_path}"
        )

    device = torch.device("cpu")
    float_model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    float_model.to(device)
    float_model.eval()

    print("Loaded float model for quantization.")

    print("Applying dynamic quantization (no calibration needed)...")

    # Specify which layer types to quantize dynamically.
    layers_to_quantize = {
        torch.nn.Conv2d,
        torch.nn.ConvTranspose2d,
        torch.nn.BatchNorm2d,
    }

    # Convert the model with a single function call
    quantized_model = torch.quantization.quantize_dynamic(
        float_model, layers_to_quantize, dtype=torch.qint8
    )

    # For dynamic quantization, we save the entire model object, not just the state_dict.
    quantized_model_path = (
        f"{config['output']['best_model_dir']}/best_model_dynamic_int8.pth"
    )
    torch.save(quantized_model, quantized_model_path)

    print(f"\nSaved dynamically quantized INT8 model to {quantized_model_path}")

    # Verify sizes
    print_model_size(float_model, "Float", path="float_temp.pth")
    print_model_size(quantized_model, "Dynamic Quantized INT8", path="quant_temp.pth")
