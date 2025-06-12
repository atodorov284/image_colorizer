from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from src.models.resnet import ResNetColorizationModel
from src.pipelines.colorization_pipeline import ColorizationPipeline


class DummyPipeline:
    """
    Dummy pipeline that generates random colorization when a real model is not available.

    This serves as a fallback when the actual model cannot be loaded or when
    demonstrating the application without a trained model.
    """

    def __init__(self) -> None:
        """Initialize the dummy pipeline with a name identifier."""
        self.name = "Dummy Colorizer"

    def predict(self, input_tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Generate random colorization as a dummy output.

        Args:
            input_tensor: Input tensor in format (3, H, W) representing grayscale image
                          replicated across three channels (LLL)

        Returns:
            torch.Tensor: Random ab values in a tensor of shape (2, H, W)
                         representing random colorization
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


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of available pre-trained models in the system.

    This function scans for available models, including the default untrained model
    and a dummy model. It also checks if there's a trained model in the checkpoints
    directory.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing model information with keys:
            - name: The model name
            - path: Path to the model weights file (if any)
            - description: Brief description of the model
            - is_trained: Boolean indicating if the model has been trained
            - is_dummy: Boolean indicating if it's a dummy model (for the dummy model only)
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


def load_model(
    model_info: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[Union[ColorizationPipeline, DummyPipeline], bool]:
    """
    Load a model based on model info and config.

    This function initializes either a real colorization model pipeline or a dummy pipeline
    based on the provided model information. It handles loading model weights if available.

    Args:
        model_info: Dictionary containing model information:
            - name: Model name
            - path: Path to model weights
            - is_trained: Whether the model has been trained
            - is_dummy: Whether to use a dummy model
        config: Configuration dictionary for the pipeline with settings for:
            - data processing
            - output paths
            - training parameters

    Returns:
        Tuple[Union[ColorizationPipeline, DummyPipeline], bool]:
            - The initialized pipeline (either real or dummy)
            - Boolean indicating if the model is trained
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
