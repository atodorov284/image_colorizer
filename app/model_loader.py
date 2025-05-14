import os
from pathlib import Path

import numpy as np
import torch

from src.models.resnet import ResNetColorizationModel
from src.pipelines.colorization_pipeline import ColorizationPipeline


class DummyPipeline:
    """Dummy pipeline for when torch is not available or no model is found."""

    def __init__(self):
        self.name = "Dummy Colorizer"

    def predict(self, input_tensor):
        """
        Return random colorization (dummy output)

        Args:
            input_tensor: Input tensor (LLL)

        Returns:
            torch-like tensor with random ab values
        """
        # Get the shape of the input and create random ab channels
        # Input is (3, H, W), output should be (2, H, W)
        if hasattr(input_tensor, "shape"):
            h, w = input_tensor.shape[1], input_tensor.shape[2]
        else:
            h, w = 256, 256

        # Create random ab channels between -1 and 1
        random_ab = np.random.uniform(-0.5, 0.5, size=(2, h, w)).astype(np.float32)

        # Convert to torch tensor
        return torch.from_numpy(random_ab)


def get_available_models():
    """
    Get a list of available pre-trained models

    Returns:
        list: List of available model names and their paths
    """
    models = []

    # Default model is always available even if untrained
    models.append(
        {
            "name": "ResNet Colorizer",
            "path": None,
            "description": "ResNet-based colorization model",
            "is_trained": False,
        }
    )

    # Add dummy model as a fallback
    models.append(
        {
            "name": "Dummy Colorizer",
            "path": None,
            "description": "Dummy colorizer that produces random colorization",
            "is_trained": False,
            "is_dummy": True,
        }
    )

    # Check for trained models
    model_dir = Path("src/models/checkpoints")
    if model_dir.exists():
        best_model_path = model_dir / "best_model.pth"
        if best_model_path.exists():
            models[0]["path"] = str(best_model_path)
            models[0]["is_trained"] = True

    return models


def load_model(model_info, config):
    """
    Load a model based on model info and config

    Args:
        model_info (dict): Model information dictionary
        config (dict): Configuration for the pipeline

    Returns:
        tuple: (model_pipeline, is_trained)
    """
    # Return dummy pipeline if model is dummy
    if model_info.get("is_dummy", False):
        print("Using dummy colorizer pipeline with random output")
        return DummyPipeline(), False

    # Otherwise load the real model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    if model_info["name"] == "ResNet Colorizer":
        model = ResNetColorizationModel()

        # Load pretrained weights if available
        if model_info["is_trained"] and model_info["path"]:
            try:
                model.load_state_dict(
                    torch.load(model_info["path"], map_location=device)
                )
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
                model_info["is_trained"] = False
    else:
        # Fallback to dummy pipeline for unknown models
        return DummyPipeline(), False

    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Create the pipeline
    try:
        pipeline = ColorizationPipeline(config, model, device)
        return pipeline, model_info["is_trained"]
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return DummyPipeline(), False
