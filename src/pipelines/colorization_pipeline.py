import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloaders.colorization_dataset import ColorizationDataset
from pipelines.base_pipeline import BasePipeline
from utils.colorization_utils import ColorizationUtils
from utils.early_stopping import EarlyStopping


class ColorizationPipeline(BasePipeline):
    def __init__(self, config: dict, model: nn.Module, device: torch.device) -> None:
        super().__init__(config, model, device)
        self.ab_bins = ColorizationUtils.get_ab_bins().to(self.device)
        self.rebalancing_weights = None
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
            num_workers=self.config.get("data", {}).get("num_workers", 0),
        )

        lambda_rebal = self.config["training"].get("lambda_rebal", 0.5)
        sigma_smooth_rebal = self.config["training"].get("sigma_smooth_rebal", 5.0)
        weights_cache_filename = f"rebal_weights_Q{self.ab_bins.shape[0]}_L{lambda_rebal}_S{sigma_smooth_rebal}.pt"
        weights_cache_path = os.path.join(
            os.path.dirname(self.config["data"]["train_cache_path"]),
            weights_cache_filename,
        )

        if os.path.exists(weights_cache_path):
            print(f"Loading rebalancing weights from cache: {weights_cache_path}")
            self.rebalancing_weights = torch.load(weights_cache_path).to(self.device)
        else:
            self.rebalancing_weights = ColorizationUtils.calculate_rebalancing_weights(
                weights_dataloader, self.ab_bins, lambda_rebal, sigma_smooth_rebal
            ).to(self.device)
            print(f"Saving rebalancing weights to cache: {weights_cache_path}")
            os.makedirs(os.path.dirname(weights_cache_path), exist_ok=True)
            torch.save(self.rebalancing_weights, weights_cache_path)

        print("Rebalancing weights initialized.")
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
            print(
                f"Using {num_samples_train} samples ({subset_percent * 100:.1f}%) from the training dataset."
            )

        num_workers = os.cpu_count() // (torch.cuda.device_count())

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config.get("data", {}).get("num_workers", num_workers),
            pin_memory=self.config.get("data", {}).get("pin_memory", True),
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config.get("data", {}).get("num_workers", num_workers),
            pin_memory=self.config.get("data", {}).get("pin_memory", True), 
            persistent_workers=True,
        )
        print(f"Train loader setup with {len(train_dataset)} images.")
        print(f"Val loader setup with {len(val_dataset)} images.")

    def setup_optimizer_criterion(self) -> None:
        self.criterion = nn.CrossEntropyLoss()# (weight=self.rebalancing_weights)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"].get("weight_decay", 10e-5),
        )
        print(
            "Optimizer (AdamW) and Criterion (CrossEntropyLoss with rebalancing) setup."
        )

    def train_epoch(self, epoch_num: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_num}", leave=False)
        for lll_inputs, ab_targets_continuous in progress_bar:
            lll_inputs = lll_inputs.to(self.device)
            target_class_indices = ColorizationUtils.ab_to_class_indices(
                ab_targets_continuous.to(self.device), self.ab_bins
            ).to(self.device)
            self.optimizer.zero_grad()
            predicted_logits = self.model(lll_inputs)
            loss = self.criterion(predicted_logits, target_class_indices)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch_num} Average Loss: {avg_epoch_loss:.6f}")
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
                target_class_indices = ColorizationUtils.ab_to_class_indices(
                    ab_targets_continuous.to(self.device), self.ab_bins
                ).to(self.device)
                predicted_logits = self.model(lll_inputs)
                loss = self.criterion(predicted_logits, target_class_indices)
                total_loss += loss.item()
                num_batches += 1

                if not visualization_done:
                    self.visualize_batch(
                        lll_inputs, ab_targets_continuous, predicted_logits
                    )
                    visualization_done = True

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Validation Average Loss: {avg_loss:.6f}")
        return avg_loss

    def visualize_batch(self, lll_inputs, ab_targets_continuous, predicted_logits):
        temperature = self.config["testing"].get("annealed_mean_temp", 0.38)
        predicted_ab_normalized = ColorizationUtils.annealed_mean_prediction(
            predicted_logits, self.ab_bins, temperature
        )

        for i in range(min(lll_inputs.size(0), 5)):
            l_input = lll_inputs[i].cpu()
            ground_truth_ab = ab_targets_continuous[i].cpu()
            predicted_ab = predicted_ab_normalized[i].cpu()

            original_l_image_input = l_input

            ground_truth_color_image = ColorizationUtils.reconstruct_image(
                original_l_image_input, ground_truth_ab
            )
            predicted_color_image = ColorizationUtils.reconstruct_image(
                original_l_image_input, predicted_ab
            )

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(original_l_image_input.permute(1, 2, 0)[:, :, 0], cmap="gray")
            ax1.set_title("Original L")
            ax1.axis("off")

            ax2.imshow(ground_truth_color_image)
            ax2.set_title("Ground Truth Color")
            ax2.axis("off")

            ax3.imshow(predicted_color_image)
            ax3.set_title("Predicted Color")
            ax3.axis("off")

            plt.show()

    def run_training(self) -> None:
        num_epochs = self.config["training"]["num_epochs"]
        model_name = self.config["model"]["name"]
        visualization_epoch_interval = self.config["training"].get(
            "visualization_epoch_interval", 50
        )
        print(f"Starting training for {num_epochs} epochs...")
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
            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
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
            # torch.save(checkpoint, ckpt_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(
                    self.best_model_dir, "best_model_rebal.pth"
                )
                torch.save(self.model.state_dict(), best_model_path)
                print(
                    f"New best model saved to {best_model_path} (Val Loss: {best_val_loss:.6f})"
                )
            if self.early_stopping(val_loss, self.model):
                print(
                    f"Early stopping triggered at epoch {epoch}. Restoring best model weights if criterion met."
                )
                if self.early_stopping.best_state:
                    self.model.load_state_dict(self.early_stopping.best_state)
                    best_early_stop_model_path = os.path.join(
                        self.best_model_dir, "best_model_earlystop_rebal.pth"
                    )
                    torch.save(self.model.state_dict(), best_early_stop_model_path)
                    print(
                        f"Best model from early stopping saved to {best_early_stop_model_path}."
                    )
                break
        print("Training finished.")

    @torch.no_grad()
    def predict(self, lll_input_tensor: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        input_batch = lll_input_tensor.to(self.device)
        predicted_logits = self.model(input_batch)
        ab_bins_for_pred = self.ab_bins.to(predicted_logits.device)
        temperature = self.config["testing"].get("annealed_mean_temp", 0.38)
        predicted_ab_normalized = ColorizationUtils.annealed_mean_prediction(
            predicted_logits, ab_bins_for_pred, temperature
        )
        return predicted_ab_normalized.cpu().squeeze(0)
