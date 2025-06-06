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
    L_SCALE: float = 100.0
    AB_SCALE: float = 128.0
    RGB_SCALE: float = 255.0
    NUM_AB_BINS: int = 313
    AB_QUANTIZATION_GRID_SIZE: int = 10
    AB_QUANTIZATION_MIN_MAX: int = 110

    @staticmethod
    def get_ab_bins() -> torch.Tensor:
        q_ab = np.load("src/resources/pts_in_hull.npy")
        return torch.from_numpy(q_ab).float()

    @staticmethod
    def ab_to_class_indices(
        ab_target: torch.Tensor, ab_bins: torch.Tensor
    ) -> torch.Tensor:
        B, _, H, W = ab_target.shape
        ab_target_denorm = ab_target * ColorizationUtils.AB_SCALE
        ab_target_denorm_flat = ab_target_denorm.permute(0, 2, 3, 1).reshape(-1, 2)
        dists_sq = torch.sum(
            (
                ab_target_denorm_flat.unsqueeze(1)
                - ab_bins.unsqueeze(0).to(ab_target_denorm_flat.device)
            )
            ** 2,
            dim=2,
        )
        class_indices = torch.argmin(dists_sq, dim=1)
        return class_indices.view(B, H, W)

    @staticmethod
    def calculate_rebalancing_weights(
        dataloader: torch.utils.data.DataLoader,
        ab_bins: torch.Tensor,
        lambda_rebal: float = 0.5,
        sigma_smooth: float = 5.0,
    ) -> torch.Tensor:
        Q = ab_bins.shape[0]
        empirical_probs = torch.zeros(Q, device=ab_bins.device)
        num_pixels = 0

        print("Calculating empirical_probs for rebalancing weights...")
        for _, ab_target_norm_batch in tqdm(dataloader):
            B, _, H, W = ab_target_norm_batch.shape
            ab_target_denorm = (
                ab_target_norm_batch.to(ab_bins.device) * ColorizationUtils.AB_SCALE
            )
            ab_target_flat = ab_target_denorm.permute(0, 2, 3, 1).reshape(B * H * W, 2)
            dists_sq = torch.sum(
                (ab_target_flat.unsqueeze(1) - ab_bins.unsqueeze(0)) ** 2, dim=2
            )
            indices = torch.argmin(dists_sq, dim=1)
            counts = torch.bincount(indices, minlength=Q)
            empirical_probs += counts.float()
            num_pixels += B * H * W

        empirical_probs /= num_pixels
        smoothed_probs = torch.from_numpy(
            gaussian_filter1d(empirical_probs.cpu().numpy(), sigma=sigma_smooth)
        ).to(ab_bins.device)
        smoothed_probs /= smoothed_probs.sum() + 1e-8
        mixed_probs = (1 - lambda_rebal) * smoothed_probs + lambda_rebal / Q
        weights = 1.0 / (mixed_probs + 1e-8)
        norm_constant = torch.sum(smoothed_probs * weights)
        weights_normalized = weights / (norm_constant + 1e-8)
        return weights_normalized

    @staticmethod
    def annealed_mean_prediction(
        predicted_logits: torch.Tensor, ab_bins: torch.Tensor, temperature: float = 0.38
    ) -> torch.Tensor:
        probabilities = torch.softmax(predicted_logits, dim=1)
        log_probs = torch.log(probabilities + 1e-20)
        annealed_log_probs = log_probs / temperature
        annealed_probs = torch.softmax(annealed_log_probs, dim=1)
        ab_bins_dev = ab_bins.to(predicted_logits.device)
        annealed_probs_rs = annealed_probs.permute(0, 2, 3, 1).unsqueeze(-1)
        ab_bins_rs = ab_bins_dev.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        predicted_ab_denorm = torch.sum(annealed_probs_rs * ab_bins_rs, dim=3)
        predicted_ab_denorm = predicted_ab_denorm.permute(0, 3, 1, 2)
        predicted_ab_normalized = predicted_ab_denorm / ColorizationUtils.AB_SCALE
        return predicted_ab_normalized

    @staticmethod
    def preprocess_image(
        pil_image: Image.Image, target_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        resize_transform = transforms.Resize(target_size)
        resized_pil = resize_transform(pil_image)
        rgb_np = np.array(resized_pil)
        rgb_float = rgb_np / ColorizationUtils.RGB_SCALE
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
        l_channel_normalized = input_tensor[0:1, :, :]
        ab_channels_normalized = predicted_ab_tensor
        l_channel_lab = l_channel_normalized * ColorizationUtils.L_SCALE
        ab_channels_lab = ab_channels_normalized * ColorizationUtils.AB_SCALE
        lab_tensor_lab = torch.cat((l_channel_lab, ab_channels_lab), dim=0)
        lab_image_np = lab_tensor_lab.detach().cpu().numpy().transpose(1, 2, 0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Conversion from CIE-LAB.*negative Z values.*",
            )
            rgb_reconstructed_float = color.lab2rgb(lab_image_np)
        return rgb_reconstructed_float
