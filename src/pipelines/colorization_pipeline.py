import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import numpy as np 
from torch.utils.data import Subset

from dataloaders.colorization_dataset import ColorizationDataset
from pipelines.base_pipeline import BasePipeline
from utils.early_stopping import EarlyStopping


class ColorizationPipeline(BasePipeline):
    """
    Pipeline for colorization using a ResNet or ViT model.
    """

    def __init__(self, config: dict, model: nn.Module, device: torch.device) -> None:
        """
        Initialize the pipeline.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        model : nn.Module
            Model to be trained.
        device : torch.device
            Device to use for training.
        """
        super().__init__(config, model, device)
        self.setup_loaders()
        self.setup_optimizer_criterion()

        self.checkpoint_dir = self.config["output"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_dir = self.config["output"]["best_model_dir"]
        os.makedirs(self.best_model_dir, exist_ok=True)

        self.early_stopping = EarlyStopping(patience=self.config["training"]["patience"], min_delta=self.config["training"]["min_delta"])

        
    def setup_loaders(self) -> None:
        """
        Sets up the data loaders, optionally with a subset of the training data.
        """
        full_dataset = ColorizationDataset(
            root_dir=self.config["data"]["train_dir"],
            captions_dir=self.config["data"]["train_captions_dir"],
            target_size=tuple(self.config["data"]["image_size"]),
            cache_path=self.config["data"]["train_cache_path"]
        )
        val_dataset = ColorizationDataset(
            root_dir=self.config["data"]["val_dir"],
            captions_dir=self.config["data"]["val_captions_dir"],
            target_size=tuple(self.config["data"]["image_size"]),
            cache_path=self.config["data"]["val_cache_path"]
        )

        # Use a subset of the training dataset if subset_percent is provided
        subset_percent = self.config["training"].get("subset_percent", 1.0)
        if 0 < subset_percent < 1.0:
            num_samples = int(len(full_dataset) * subset_percent)
            indices = np.random.choice(len(full_dataset), num_samples, replace=False)
            full_dataset = Subset(full_dataset, indices)
            print(f"Using {num_samples} samples ({subset_percent*100:.1f}%) from the training dataset.")

        self.train_loader = DataLoader(
            full_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
        )

        print(f"Train loader setup with {len(full_dataset)} images.")
        print(f"Val loader setup with {len(val_dataset)} images.")

    def setup_optimizer_criterion(self) -> None:
        """
        Sets up the optimizer and criterion.
        """
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config["training"]["learning_rate"]
        )
        print("Optimizer (Adam) and Criterion (MSELoss) setup.")

    def train_epoch(self, epoch_num: int) -> float:
        """
        Runs one epoch of training.

        Parameters
        ----------
        epoch_num : int
            The current epoch number.

        Returns
        -------
        float
            The average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_num}", leave=False)

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

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch_num} Average Loss: {avg_epoch_loss:.6f}")

        return avg_epoch_loss

    def evaluate(self) -> None:
        """
        Evaluates the model on the validation set.
        """

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Evaluating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Validation Average Loss: {avg_loss:.6f}")
        return avg_loss

    def run_training(self) -> None:
        """
        Runs the full training loop with early stopping.
        """
        num_epochs = self.config["training"]["num_epochs"]
        model_name = self.config["model"]["name"]
        print(f"Starting training for {num_epochs} epochs...")

        start_time = datetime.now()
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            end_time = datetime.now()
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'wall_time': (end_time - start_time).total_seconds()
            }
            ckpt_name = f"{model_name}_epoch_{epoch:03d}.pth"
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
            torch.save(checkpoint, ckpt_path)

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch}. Restoring best model weights.")
                self.early_stopping.restore_best_weights(self.model)
                
                best_model_path = os.path.join(self.best_model_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}.")
                break


        print("Training finished.")

    @torch.no_grad()
    def predict(self, lll_input_tensor: torch.Tensor) -> torch.Tensor:
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
        self.model.eval()
        input_batch = lll_input_tensor.unsqueeze(0).to(self.device)
        predicted_ab = self.model(input_batch).squeeze(0).cpu()
        return predicted_ab
