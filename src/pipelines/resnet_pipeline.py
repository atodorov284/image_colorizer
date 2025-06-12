import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datetime import datetime

from dataloaders.colorization_dataset import ColorizationDataset
from pipelines.base_pipeline import BasePipeline
from utils.early_stopping import EarlyStopping

from utils.predicting_utils import PredictingUtils

import os


class ResNetPipeline(BasePipeline):
    """Pipeline for ResNet-based colorization models."""

    def __init__(self, config: dict, model: nn.Module, device: torch.device) -> None:
        """Initialize the ResNet pipeline."""
        super().__init__(config, model, device)

        if torch.cuda.is_available():
            self.num_workers = os.cpu_count() // torch.cuda.device_count()
        else:
            self.num_workers = os.cpu_count() // 2
        print(f"Resnet Pipeline - Workers: {self.num_workers}")

        self.early_stopping = EarlyStopping(
            patience=self.config["training"]["patience"],
            min_delta=self.config["training"]["min_delta"],
        )
        self.setup_loaders()
        self.setup_optimizer_criterion()

    def setup_loaders(self) -> None:
        """Setup data loaders for ResNet training."""
        # Training dataset
        train_dataset = ColorizationDataset(
            root_dir=self.config["data"]["train_dir"],
            captions_dir=self.config["data"]["train_captions_dir"],
            target_size=tuple(self.config["data"]["image_size"]),
            cache_path=self.config["data"]["train_cache_path"],
        )

        # Validation dataset
        val_dataset = ColorizationDataset(
            root_dir=self.config["data"]["val_dir"],
            captions_dir=self.config["data"]["val_captions_dir"],
            target_size=tuple(self.config["data"]["image_size"]),
            cache_path=self.config["data"]["val_cache_path"],
        )

        # Optional subset
        subset_percent = self.config["training"].get("subset_percent", 1.0)
        if 0 < subset_percent < 1.0:
            num_samples = int(len(train_dataset) * subset_percent)
            indices = np.random.choice(len(train_dataset), num_samples, replace=False)
            train_dataset = Subset(train_dataset, indices)
            print(
                f"Using {num_samples} samples ({subset_percent * 100:.1f}%) from training dataset."
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        print(f"ResNet Pipeline - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    def setup_optimizer_criterion(self) -> None:
        """Setup optimizer and loss criterion for ResNet."""
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config["training"]["learning_rate"]
        )
        print("ResNet Pipeline - Using Adam optimizer and MSE loss")

    def train_epoch(self, epoch_num: int) -> float:
        """Train one epoch for ResNet."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader, desc=f"ResNet Epoch {epoch_num}", leave=False
        )
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def evaluate(self, **kwargs) -> float:
        """Evaluate ResNet model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                self.val_loader, desc="ResNet Eval", leave=False
            ):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def run_training(self) -> None:
        """Run ResNet training loop."""
        num_epochs = self.config["training"]["num_epochs"]
        model_name = self.config["model"]["name"]
        print(f"Starting ResNet training for {num_epochs} epochs...")

        start_time = datetime.now()
        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            print(
                f"ResNet Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}"
            )

            # Save checkpoint
            end_time = datetime.now()
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "wall_time": (end_time - start_time).total_seconds(),
            }
            ckpt_name = f"{model_name}_epoch_{epoch:03d}.pth"
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
            torch.save(checkpoint, ckpt_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.best_model_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)

            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"ResNet early stopping at epoch {epoch}")
                if self.early_stopping.best_state:
                    self.model.load_state_dict(self.early_stopping.best_state)
                    best_early_model_path = os.path.join(
                        self.best_model_dir, "best_model_earlystop.pth"
                    )
                    torch.save(self.model.state_dict(), best_early_model_path)
                break

        print("ResNet training finished.")

    @torch.no_grad()
    def predict(self, lll_input_tensor: torch.Tensor) -> torch.Tensor:
        """Predict with ResNet model."""
        return PredictingUtils.predict_resnet(self.model, self.device, lll_input_tensor)
