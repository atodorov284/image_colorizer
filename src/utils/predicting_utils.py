import os
from typing import List

import torch
from PIL import Image
from skimage.color import lab2rgb


class PredictingUtils:
    """
    A utility class for loading a model and predicting AB channels from LLL input tensors.
    """

    @torch.no_grad()
    def predict(model, device, lll_input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predicts AB channels for a single LLL input tensor.

        Parameters
        ----------
        lll_input_tensor : torch.Tensor
            Input tensor (LLL).

        Returns
        -------
        torch.Tensor
            Predicted AB channels tensor.
        """
        model.eval()
        input_batch = lll_input_tensor.unsqueeze(0).to(device)
        predicted_ab = model(input_batch).squeeze(0).cpu()
        return predicted_ab

    def collect_images(test_dir: str, subset_percent: float) -> List[str]:
        """
        Collects image paths from a directory.

        Parameters
        ----------
        test_dir : str
            Directory containing images.
        subset_percent : float
            Percentage of images to collect.

        Returns
        -------
        List[str]
            List of image paths.
        """
        img_paths = []
        if os.path.isdir(test_dir):
            img_paths = [
                os.path.join(test_dir, f)
                for f in os.listdir(test_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        elif os.path.isfile(test_dir):
            return [test_dir]
        else:
            raise FileNotFoundError(
                f"Invalid test_dir: {test_dir}. It should be a file or a directory."
            )
        if subset_percent < 1.0:
            img_paths = img_paths[: int(len(img_paths) * subset_percent)]
        return img_paths
