import os
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.colorization_utils import ColorizationUtils


class ColorizationDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing images for colorization.

    Args:
        root_dir (str): Path to the directory containing images.
        target_size (Tuple[int, int], optional): Target size (height, width)
                                                 for resizing images. Defaults to (256, 256).
    """

    def __init__(
        self, root_dir: str, target_size: Tuple[int, int] = (256, 256)
    ) -> None:
        """Constructor for ColorizationDataset."""
        self.root_dir: str = root_dir
        self.target_size: Tuple[int, int] = target_size
        self.image_files: list[str] = [
            f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))
        ]

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads, resizes, and preprocesses a single image from the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input (LLL) and Target (AB) tensors.
        """
        img_path: str = os.path.join(self.root_dir, self.image_files[idx])
        img_rgb_pil: Image.Image = Image.open(img_path).convert("RGB")
        input_tensor, output_tensor = ColorizationUtils.preprocess_image(
            img_rgb_pil, self.target_size
        )
        return input_tensor, output_tensor
