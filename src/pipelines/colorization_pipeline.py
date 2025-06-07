import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloaders.colorization_dataset import ColorizationDataset
from pipelines.base_pipeline import BasePipeline
from utils.colorization_utils import ColorizationUtils
from utils.early_stopping import EarlyStopping
from utils.predicting_utils import PredictingUtils


class ColorizationPipeline(BasePipeline):
    def __init__(self, config: dict, model: nn.Module, device: torch.device) -> None:
        super().__init__(config, model, device)
        if torch.cuda.is_available():
            self.num_workers = os.cpu_count() // (torch.cuda.device_count())
        else:
            self.num_workers = os.cpu_count() // 2
        print(f"Number of workers: {self.num_workers}")
        self.ab_bins = ColorizationUtils.get_ab_bins().to(self.device)
        self.rebalancing_weights = None
        self.scheduler = None
        self.setup_loaders()
        self.setup_optimizer_criterion()
        self.checkpoint_dir = self.config["output"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_dir = self.config["output"]["best_model_dir"]
        os.makedirs(self.best_model_dir, exist_ok=True)
        self.early_stopping = EarlyStopping(
            patience=self.config["training"]["patience"],
            min_delta=self.config["training"]["min_delta"],
        )

    def setup_loaders(self) -> None:
        full_train_dataset_for_weights = ColorizationDataset(
            root_dir=self.config["data"]["train_dir"],
            captions_dir=self.config["data"]["train_captions_dir"],
            target_size=tuple(self.config["data"]["image_size"]),
            cache_path=self.config["data"]["train_cache_path"],
        )

        weights_dataloader = DataLoader(
            full_train_dataset_for_weights,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.num_workers,
        )

        lambda_rebal = self.config["training"].get("lambda_rebal", 0.5)
        sigma_smooth_rebal = self.config["training"].get("sigma_smooth_rebal", 5.0)
        weights_cache_filename = f"rebal_weights_Q{self.ab_bins.shape[0]}_L{lambda_rebal}_S{sigma_smooth_rebal}.pt"
        weights_cache_path = os.path.join(
            os.path.dirname(self.config["data"]["train_cache_path"]),
            weights_cache_filename,
        )

        if os.path.exists(weights_cache_path):
            self.rebalancing_weights = torch.load(
                weights_cache_path, map_location=self.device
            )
        else:
            self.rebalancing_weights = ColorizationUtils.calculate_rebalancing_weights(
                weights_dataloader, self.ab_bins, lambda_rebal, sigma_smooth_rebal
            ).to(self.device)
            os.makedirs(os.path.dirname(weights_cache_path), exist_ok=True)
            torch.save(self.rebalancing_weights, weights_cache_path)

        train_dataset = ColorizationDataset(
            root_dir=self.config["data"]["train_dir"],
            captions_dir=self.config["data"]["train_captions_dir"],
            target_size=tuple(self.config["data"]["image_size"]),
            cache_path=self.config["data"]["train_cache_path"],
        )
        val_dataset = ColorizationDataset(
            root_dir=self.config["data"]["val_dir"],
            captions_dir=self.config["data"]["val_captions_dir"],
            target_size=tuple(self.config["data"]["image_size"]),
            cache_path=self.config["data"]["val_cache_path"],
        )

        subset_percent = self.config["training"].get("subset_percent", 1.0)
        if 0 < subset_percent < 1.0:
            num_samples_train = int(len(train_dataset) * subset_percent)
            indices_train = np.random.choice(
                len(train_dataset), num_samples_train, replace=False
            )
            train_dataset = Subset(train_dataset, indices_train)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.config.get("data", {}).get("pin_memory", True),
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.config.get("data", {}).get("pin_memory", True),
            persistent_workers=True,
        )

    def setup_optimizer_criterion(self) -> None:
        # Use L1 loss for ab regression
        self.criterion = nn.L1Loss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"].get("weight_decay", 10e-5),
        )
        if "lr_scheduler" in self.config["training"]:
            scheduler_config = self.config["training"]["lr_scheduler"]
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"],
            )

    def train_epoch(self, epoch_num: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_num}", leave=False)
        for lll_inputs, ab_targets_continuous in progress_bar:
            lll_inputs = lll_inputs.to(self.device)
            ab_targets_continuous = ab_targets_continuous.to(self.device)
            self.optimizer.zero_grad()
            ab_pred = self.model(lll_inputs)
            # Compute per-pixel L1 loss
            per_pixel_loss = torch.abs(ab_pred - ab_targets_continuous).sum(
                dim=1
            )  # [B, H, W]
            # Get ab class indices for each pixel
            with torch.no_grad():
                ab_class_indices = ColorizationUtils.ab_to_class_indices(
                    ab_targets_continuous, self.ab_bins
                )  # [B, H, W]
            # Gather weights for each pixel
            pixel_weights = self.rebalancing_weights[ab_class_indices]  # [B, H, W]
            # Weighted loss
            weighted_loss = (per_pixel_loss * pixel_weights).sum() / pixel_weights.sum()
            loss = weighted_loss
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return avg_epoch_loss

    def evaluate(self, visualize: bool = False) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        visualization_done = not visualize
        with torch.no_grad():
            for lll_inputs, ab_targets_continuous in tqdm(
                self.val_loader, desc="Evaluating", leave=False
            ):
                lll_inputs = lll_inputs.to(self.device)
                ab_targets_continuous = ab_targets_continuous.to(self.device)
                ab_pred = self.model(lll_inputs)
                per_pixel_loss = torch.abs(ab_pred - ab_targets_continuous).sum(
                    dim=1
                )  # [B, H, W]
                ab_class_indices = ColorizationUtils.ab_to_class_indices(
                    ab_targets_continuous, self.ab_bins
                )  # [B, H, W]
                pixel_weights = self.rebalancing_weights[ab_class_indices]  # [B, H, W]
                weighted_loss = (
                    per_pixel_loss * pixel_weights
                ).sum() / pixel_weights.sum()
                loss = weighted_loss
                total_loss += loss.item()
                num_batches += 1
                if not visualization_done:
                    self.visualize_batch(lll_inputs, ab_targets_continuous, ab_pred)
                    visualization_done = True
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def visualize_batch(self, lll_inputs, ab_targets_continuous, ab_pred):
        for i in range(min(lll_inputs.size(0), 5)):
            l_input = lll_inputs[i].cpu()
            ground_truth_ab = ab_targets_continuous[i].cpu()
            predicted_ab = ab_pred[i].cpu()
            ground_truth_color_image = ColorizationUtils.reconstruct_image(
                l_input, ground_truth_ab
            )
            predicted_color_image = ColorizationUtils.reconstruct_image(
                l_input, predicted_ab
            )
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(l_input.permute(1, 2, 0)[:, :, 0], cmap="gray")
            ax1.axis("off")
            ax2.imshow(ground_truth_color_image)
            ax2.axis("off")
            ax3.imshow(predicted_color_image)
            ax3.axis("off")
            plt.show()

    def run_training(self) -> None:
        num_epochs = self.config["training"]["num_epochs"]
        model_name = self.config["model"]["name"]
        visualization_epoch_interval = self.config["training"].get(
            "visualization_epoch_interval", 50
        )
        start_time = datetime.now()
        best_val_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            should_visualize = (
                epoch == 1
                or epoch % visualization_epoch_interval == 0
                or epoch == num_epochs
            )
            val_loss = self.evaluate(visualize=should_visualize)

            if self.scheduler:
                self.scheduler.step()

            end_time = datetime.now()
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "wall_time": (end_time - start_time).total_seconds(),
                "ab_bins": self.ab_bins.cpu(),
                "rebalancing_weights": self.rebalancing_weights.cpu(),
            }
            ckpt_name = f"{model_name}_epoch_{epoch:03d}_rebal.pth"
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(
                    self.best_model_dir, "best_model_rebal.pth"
                )
                torch.save(self.model.state_dict(), best_model_path)

            if self.early_stopping(val_loss, self.model):
                if self.early_stopping.best_state:
                    self.model.load_state_dict(self.early_stopping.best_state)
                    best_early_stop_model_path = os.path.join(
                        self.best_model_dir, "best_model_earlystop_rebal.pth"
                    )
                    torch.save(self.model.state_dict(), best_early_stop_model_path)

                break

    @torch.no_grad()
    def predict(self, input_image: Image.Image) -> np.ndarray:
        """
        Predict the colorized image from a PIL input using PredictingUtils and the current model.
        Args:
            input_image (Image.Image): Input RGB image.
        Returns:
            np.ndarray: Colorized RGB image as a numpy array in [0, 1].
        """
        return PredictingUtils.predict(self.model, self.device, input_image)
