from typing import Tuple

import numpy as np
import torch
from PIL import Image
from skimage import color
from torchvision import transforms


class ColorizationUtils:
    """Utility class for image colorization preprocessing and reconstruction."""

    L_SCALE: float = 100.0
    AB_SCALE: float = 128.0
    RGB_SCALE: float = 255.0

    @staticmethod
    def preprocess_image(
        pil_image: Image.Image, target_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resizes and preprocesses a PIL image into input (LLL) and target (AB) tensors.

        Args:
            pil_image (Image.Image): Input PIL RGB image.
            target_size (Tuple[int, int]): Target (height, width) for resizing.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - input_tensor (torch.Tensor): Normalized L channel repeated 3 times (LLL).
                                               Shape [3, H, W], Range [0, 1].
                - output_tensor (torch.Tensor): Normalized AB channels.
                                                Shape [2, H, W], Range [-1, 1].
        """
        resize_transform = transforms.Resize(target_size)
        resized_pil = resize_transform(pil_image)
        rgb_np = np.array(resized_pil)
        rgb_float = rgb_np / ColorizationUtils.RGB_SCALE
        lab_image = color.rgb2lab(rgb_float)  # Output shape (H, W, 3)

        l_channel = lab_image[:, :, 0:1]  # Shape (H, W, 1)
        l_normalized = l_channel / ColorizationUtils.L_SCALE  # Range [0, 1]

        lll_normalized = np.concatenate([l_normalized] * 3, axis=2)  # Shape (H, W, 3)
        input_tensor = (
            torch.from_numpy(lll_normalized).permute(2, 0, 1).float()
        )  # Shape (3, H, W)

        ab_channels = lab_image[:, :, 1:3]  # Shape (H, W, 2)
        ab_normalized = ab_channels / ColorizationUtils.AB_SCALE  # Range [-1, 1] approx
        output_tensor = (
            torch.from_numpy(ab_normalized).permute(2, 0, 1).float()
        )  # Shape (2, H, W)

        return input_tensor, output_tensor

    @staticmethod
    def reconstruct_image(
        input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Reconstructs an RGB image from normalized LLL input and normalized AB target tensors.

        Args:
            input_tensor (torch.Tensor): Normalized LLL tensor.
                                         Shape [3, H, W], Range [0, 1].
            target_tensor (torch.Tensor): Normalized AB tensor.
                                          Shape [2, H, W], Range [-1, 1].

        Returns:
            np.ndarray: Reconstructed RGB image as a NumPy array.
                        Shape [H, W, 3], Range [0, 1], dtype float64 or float32.
        """
        l_channel_normalized = input_tensor[0:1, :, :]  # Shape (1, H, W)
        ab_channels_normalized = target_tensor  # Shape (2, H, W)

        l_channel_lab = (
            l_channel_normalized * ColorizationUtils.L_SCALE
        )  # Range [0, 100]
        ab_channels_lab = (
            ab_channels_normalized * ColorizationUtils.AB_SCALE
        )  # Range [-128, 128]

        lab_tensor_lab = torch.cat(
            (l_channel_lab, ab_channels_lab), dim=0
        )  # Shape (3, H, W)

        lab_image_np = (
            lab_tensor_lab.detach().cpu().numpy().transpose(1, 2, 0)
        )  # Shape (H, W, 3)

        rgb_reconstructed_float = color.lab2rgb(
            lab_image_np
        )  # Output shape (H, W, 3), float range [0, 1]

        return rgb_reconstructed_float
