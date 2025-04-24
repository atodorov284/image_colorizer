import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage import color
from torch.utils.data import DataLoader, Dataset
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


if __name__ == "__main__":
    data_folder: str = "data"
    train_dir: str = os.path.join(data_folder, "train2017")

    TARGET_IMAGE_SIZE: Tuple[int, int] = (256, 256)
    BATCH_SIZE: int = 32

    print(f"Creating dataset from: {train_dir}")
    train_dataset: ColorizationDataset = ColorizationDataset(
        root_dir=train_dir, target_size=TARGET_IMAGE_SIZE
    )

    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    print(f"\nTotal training images found: {len(train_dataset)}")

    print("\nIterating through DataLoader (first batch):")
    for i, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {i + 1}")
        print(f"  Input batch shape: {inputs.shape}")
        print(f"  Target batch shape: {targets.shape}")

        if i == 0:
            reconstructed_rgb_image: np.ndarray = ColorizationUtils.reconstruct_image(
                inputs[0], targets[0]
            )

            input_display: np.ndarray = (
                inputs[0].detach().cpu().permute(1, 2, 0).numpy()
            )

            print(f"  Input display shape: {input_display.shape}")
            print(f"  Reconstructed RGB shape: {reconstructed_rgb_image.shape}")

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(input_display)
            plt.title("Input Grayscale (L channel)")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_rgb_image)
            plt.title("Reconstructed Color Image")
            plt.axis("off")

            plt.suptitle(f"Reconstruction Example (Batch {i + 1}, Item 0)")
            plt.tight_layout()
            plt.show()

            break
