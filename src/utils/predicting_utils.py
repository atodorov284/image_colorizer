import os
from typing import List, Tuple, Union

import numpy as np
import skimage.color as color
import torch
import torch.nn.functional as F
from PIL import Image

from utils.colorization_utils import ColorizationUtils


class PredictingUtils:
    """
    Utility class for loading a model and predicting ab channels from L channel input tensors.
    Provides image collection, loading, resizing, preprocessing, and postprocessing utilities.
    """

    @staticmethod
    @torch.no_grad()
    def predict_vgg(
        model: torch.nn.Module,
        device: torch.device,
        input_image: Image.Image,
        input_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Predict the ab channels for a given input image using the provided model.
        Args:
            model (torch.nn.Module): Trained colorization model.
            device (torch.device): Device to run the model on.
            input_image (Image.Image): Input RGB image.
        Returns:
            np.ndarray: Colorized RGB image as a numpy array in [0, 1].
        """
        l_orig_tensor, l_resized_tensor = PredictingUtils.preprocess_img(
            input_image, target_hw=input_size
        )
        l_resized_tensor = l_resized_tensor.to(device)
        ab_predicted = model(l_resized_tensor).cpu()
        colorized_img = PredictingUtils.postprocess_tens(l_orig_tensor, ab_predicted)
        return colorized_img

    @staticmethod
    @torch.no_grad()
    def predict_resnet(
        model: torch.nn.Module,
        device: torch.device,
        input_image: Image.Image,
        input_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Predicts AB channels for a single LLL input tensor.

        Parameters
        ----------
        lll_input_tensor : torch.Tensor
            Input tensor (LLL).

        Returns
        -------
        np.ndarray
            The colorized image
        """
        model.eval()
        lll, _ = ColorizationUtils.preprocess_image(input_image, input_size)
        input_batch = lll.unsqueeze(0).to(device)
        predicted_ab = model(input_batch).squeeze(0).cpu()

        l_orig_tensor, l_resized_tensor = PredictingUtils.preprocess_img(
            input_image, target_hw=(256, 256)
        )
        predicted_ab *= ColorizationUtils.AB_SCALE
        colorized_img = PredictingUtils.postprocess_tens(
            l_orig_tensor, predicted_ab.unsqueeze(0)
        )

        return colorized_img

    @staticmethod
    def collect_images(image_dir: str, subset_percent: float) -> List[str]:
        """
        Collect image file paths from a directory or a single file.
        Args:
            image_dir (str): Directory or file path containing images.
            subset_percent (float): Percentage of images to collect (0 < subset_percent <= 1).
        Returns:
            List[str]: List of image file paths.
        Raises:
            FileNotFoundError: If the path is invalid.
        """
        image_paths: List[str] = []
        if os.path.isdir(image_dir):
            image_paths = [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        elif os.path.isfile(image_dir):
            return [image_dir]
        else:
            raise FileNotFoundError(
                f"Invalid image_dir: {image_dir}. It should be a file or a directory."
            )
        if subset_percent < 1.0:
            image_paths = image_paths[: int(len(image_paths) * subset_percent)]
        return image_paths

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load an image from a file path as a numpy array (always 3 channels).
        Args:
            image_path (str): Path to the image file.
        Returns:
            np.ndarray: Loaded image as a numpy array (H, W, 3).
        """
        image_np = np.asarray(Image.open(image_path))
        if image_np.ndim == 2:
            image_np = np.tile(image_np[:, :, None], 3)
        return image_np

    @staticmethod
    def resize_image(
        image: np.ndarray, target_hw: Tuple[int, int] = (256, 256), resample: int = 3
    ) -> np.ndarray:
        """
        Resize an image to the target height and width.
        Args:
            image (np.ndarray): Input image array.
            target_hw (Tuple[int, int]): Target (height, width).
            resample (int): Resampling filter.
        Returns:
            np.ndarray: Resized image array.
        """
        return np.asarray(
            Image.fromarray(image).resize(
                (target_hw[1], target_hw[0]), resample=resample
            )
        )

    @staticmethod
    def preprocess_img(
        image_rgb: Union[np.ndarray, Image.Image],
        target_hw: Tuple[int, int] = (256, 256),
        resample: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess an RGB image for model input: resize, convert to LAB, extract L channel.
        Args:
            image_rgb (Union[np.ndarray, Image.Image]): Input RGB image.
            target_hw (Tuple[int, int]): Target (height, width).
            resample (int): Resampling filter.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Original size L channel tensor (1, 1, H_orig, W_orig)
                - Resized L channel tensor (1, 1, H, W)
        """
        if isinstance(image_rgb, Image.Image):
            image_rgb = np.asarray(image_rgb)
        image_rgb_resized = PredictingUtils.resize_image(image_rgb, target_hw, resample)
        lab_orig = color.rgb2lab(image_rgb)
        lab_resized = color.rgb2lab(image_rgb_resized)
        l_orig = lab_orig[:, :, 0]
        l_resized = lab_resized[:, :, 0]
        l_orig_tensor = torch.tensor(l_orig, dtype=torch.float32)[None, None, :, :]
        l_resized_tensor = torch.tensor(l_resized, dtype=torch.float32)[
            None, None, :, :
        ]
        return l_orig_tensor, l_resized_tensor

    @staticmethod
    def postprocess_tens(
        l_orig_tensor: torch.Tensor, ab_predicted: torch.Tensor, mode: str = "bilinear"
    ) -> np.ndarray:
        """
        Postprocess model output to reconstruct a colorized RGB image.
        Args:
            l_orig_tensor (torch.Tensor): Original size L channel tensor (1, 1, H_orig, W_orig).
            ab_predicted (torch.Tensor): Predicted ab channels (1, 2, H, W).
            mode (str): Interpolation mode for resizing ab channels.
        Returns:
            np.ndarray: Colorized RGB image as a numpy array in [0, 1].
        """
        orig_hw = l_orig_tensor.shape[2:]
        pred_hw = ab_predicted.shape[2:]
        if orig_hw != pred_hw:
            ab_predicted_resized = F.interpolate(ab_predicted, size=orig_hw, mode=mode)
        else:
            ab_predicted_resized = ab_predicted
        lab_tensor = torch.cat((l_orig_tensor, ab_predicted_resized), dim=1)
        lab_np = lab_tensor.data.cpu().numpy()[0, ...].transpose((1, 2, 0))
        return color.lab2rgb(lab_np)
