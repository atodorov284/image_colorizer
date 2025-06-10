import warnings
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from skimage import color
from torchvision import transforms
from tqdm import tqdm


class ColorizationUtils:
    """
    Utility class for colorization-related operations, including normalization, quantization,
    rebalancing, and image preprocessing for colorization models.
    """

    L_SCALE: float = 100.0
    L_CENT: float = 50.0
    AB_SCALE: float = 128.0
    RGB_SCALE: float = 255.0
    NUM_AB_BINS: int = 313
    AB_QUANTIZATION_GRID_SIZE: float = 10.0
    AB_QUANTIZATION_MIN_MAX: float = 110.0

    @staticmethod
    def normalize_l_channel(l_channel: torch.Tensor) -> torch.Tensor:
        """
        Normalize the L channel of a LAB image.
        Args:
            l_channel (torch.Tensor): Luminance channel tensor.
        Returns:
            torch.Tensor: Normalized L channel.
        """
        return (l_channel - ColorizationUtils.L_CENT) / ColorizationUtils.L_SCALE

    @staticmethod
    def unnormalize_l_channel(l_channel_normalized: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize the L channel of a LAB image.
        Args:
            l_channel_normalized (torch.Tensor): Normalized L channel tensor.
        Returns:
            torch.Tensor: Unnormalized L channel.
        """
        return (
            l_channel_normalized * ColorizationUtils.L_SCALE + ColorizationUtils.L_CENT
        )

    @staticmethod
    def normalize_ab_channels(ab_channels: torch.Tensor) -> torch.Tensor:
        """
        Normalize the ab channels of a LAB image.
        Args:
            ab_channels (torch.Tensor): ab channels tensor.
        Returns:
            torch.Tensor: Normalized ab channels.
        """
        return ab_channels / ColorizationUtils.AB_SCALE

    @staticmethod
    def unnormalize_ab_channels(ab_channels_normalized: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize the ab channels of a LAB image.
        Args:
            ab_channels_normalized (torch.Tensor): Normalized ab channels tensor.
        Returns:
            torch.Tensor: Unnormalized ab channels.
        """
        return ab_channels_normalized * ColorizationUtils.AB_SCALE

    @staticmethod
    def get_ab_bins() -> torch.Tensor:
        """
        Load quantized ab bins from a numpy file.
        Returns:
            torch.Tensor: Quantized ab bins (Q x 2).
        """
        ab_bins = np.load("src/resources/pts_in_hull.npy")
        return torch.from_numpy(ab_bins).float()

    @staticmethod
    def ab_to_class_indices(
        ab_channels: torch.Tensor, ab_bins: torch.Tensor
    ) -> torch.Tensor:
        """
        Map ab channels to nearest quantized bin indices.
        Args:
            ab_channels (torch.Tensor): Normalized ab channels (B, 2, H, W).
            ab_bins (torch.Tensor): Quantized ab bins (Q, 2).
        Returns:
            torch.Tensor: Class indices (B, H, W).
        """
        batch_size, _, height, width = ab_channels.shape
        ab_channels_denorm = ab_channels * ColorizationUtils.AB_SCALE
        ab_channels_flat = ab_channels_denorm.permute(0, 2, 3, 1).reshape(-1, 2)
        dists_sq = torch.sum(
            (
                ab_channels_flat.unsqueeze(1)
                - ab_bins.unsqueeze(0).to(ab_channels_flat.device)
            )
            ** 2,
            dim=2,
        )
        class_indices = torch.argmin(dists_sq, dim=1)
        return class_indices.view(batch_size, height, width)

    @staticmethod
    def calculate_rebalancing_weights(
        dataloader: torch.utils.data.DataLoader,
        ab_bins: torch.Tensor,
        lambda_rebal: float = 0.5,
        sigma_smooth: float = 5.0,
    ) -> torch.Tensor:
        """
        Calculate rebalancing weights for quantized ab bins based on empirical distribution.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader yielding normalized ab channels.
            ab_bins (torch.Tensor): Quantized ab bins (Q, 2).
            lambda_rebal (float): Mixing parameter for rebalancing.
            sigma_smooth (float): Smoothing parameter for empirical distribution.
        Returns:
            torch.Tensor: Normalized rebalancing weights (Q,).
        """
        num_bins = ab_bins.shape[0]
        empirical_probs = torch.zeros(num_bins, device=ab_bins.device)
        total_pixels = 0

        print("Calculating empirical probabilities for rebalancing weights...")
        for _, ab_channels_batch in tqdm(dataloader):
            batch_size, _, height, width = ab_channels_batch.shape
            ab_channels_denorm = (
                ab_channels_batch.to(ab_bins.device) * ColorizationUtils.AB_SCALE
            )
            ab_channels_flat = ab_channels_denorm.permute(0, 2, 3, 1).reshape(
                batch_size * height * width, 2
            )
            dists_sq = torch.sum(
                (ab_channels_flat.unsqueeze(1) - ab_bins.unsqueeze(0)) ** 2, dim=2
            )
            indices = torch.argmin(dists_sq, dim=1)
            counts = torch.bincount(indices, minlength=num_bins)
            empirical_probs += counts.float()
            total_pixels += batch_size * height * width

        empirical_probs /= total_pixels
        smoothed_probs = torch.from_numpy(
            gaussian_filter1d(empirical_probs.cpu().numpy(), sigma=sigma_smooth)
        ).to(ab_bins.device)
        smoothed_probs /= smoothed_probs.sum() + 1e-8
        mixed_probs = (1 - lambda_rebal) * smoothed_probs + lambda_rebal / num_bins
        weights = 1.0 / (mixed_probs + 1e-8)
        norm_constant = torch.sum(smoothed_probs * weights)
        weights_normalized = weights / (norm_constant + 1e-8)
        return weights_normalized

    @staticmethod
    def annealed_mean_prediction(
        logits: torch.Tensor, ab_bins: torch.Tensor, temperature: float = 0.38
    ) -> torch.Tensor:
        """
        Compute the annealed mean prediction for ab channels from logits.
        Args:
            logits (torch.Tensor): Predicted logits (B, Q, H, W).
            ab_bins (torch.Tensor): Quantized ab bins (Q, 2).
            temperature (float): Softmax temperature for annealing.
        Returns:
            torch.Tensor: Predicted ab channels (B, 2, H, W), normalized.
        """
        probabilities = torch.softmax(logits, dim=1)
        log_probs = torch.log(probabilities + 1e-20)
        annealed_log_probs = log_probs / temperature
        annealed_probs = torch.softmax(annealed_log_probs, dim=1)
        ab_bins_device = ab_bins.to(logits.device)
        annealed_probs_rs = annealed_probs.permute(0, 2, 3, 1).unsqueeze(-1)
        ab_bins_rs = ab_bins_device.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        predicted_ab_denorm = torch.sum(annealed_probs_rs * ab_bins_rs, dim=3)
        predicted_ab_denorm = predicted_ab_denorm.permute(0, 3, 1, 2)
        predicted_ab_normalized = predicted_ab_denorm / ColorizationUtils.AB_SCALE
        return predicted_ab_normalized

    @staticmethod
    def preprocess_image(
        image: Image.Image, target_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess a PIL image for colorization: resize, convert to LAB, normalize.
        Args:
            image (Image.Image): Input RGB image.
            target_size (Tuple[int, int]): Target size (height, width).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Input tensor (3, H, W) with normalized L channel repeated in all channels.
                - Output tensor (2, H, W) with normalized ab channels.
        """
        resize_transform = transforms.Resize(target_size)
        resized_image = resize_transform(image)
        rgb_array = np.array(resized_image)
        rgb_float = rgb_array / ColorizationUtils.RGB_SCALE
        lab_image = color.rgb2lab(rgb_float)
        l_channel = lab_image[:, :, 0:1]
        l_normalized = l_channel / ColorizationUtils.L_SCALE
        lll_normalized = np.concatenate([l_normalized] * 3, axis=2)
        input_tensor = torch.from_numpy(lll_normalized).permute(2, 0, 1).float()
        ab_channels = lab_image[:, :, 1:3]
        ab_normalized = ab_channels / ColorizationUtils.AB_SCALE
        output_tensor = torch.from_numpy(ab_normalized).permute(2, 0, 1).float()
        return input_tensor, output_tensor

    @staticmethod
    def reconstruct_image(
        input_tensor: torch.Tensor, predicted_ab_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Reconstruct an RGB image from normalized L and ab tensors.
        Args:
            input_tensor (torch.Tensor): Input tensor (3, H, W) with normalized L channel.
            predicted_ab_tensor (torch.Tensor): Predicted ab channels (2, H, W), normalized.
        Returns:
            np.ndarray: Reconstructed RGB image as float array in [0, 1].
        """
        l_channel_normalized = input_tensor[0:1, :, :]
        ab_channels_normalized = predicted_ab_tensor
        l_channel_lab = l_channel_normalized * ColorizationUtils.L_SCALE
        ab_channels_lab = ab_channels_normalized * ColorizationUtils.AB_SCALE
        lab_tensor = torch.cat((l_channel_lab, ab_channels_lab), dim=0)
        lab_image_np = lab_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Conversion from CIE-LAB.*negative Z values.*",
            )
            rgb_reconstructed = color.lab2rgb(lab_image_np)
        return rgb_reconstructed
