import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.colorization_dataset import ColorizationDataset
from pipelines.base_pipeline import BasePipeline


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
        # Pass the specific model instance (ResNet or ViT)
        super().__init__(config, model, device)
        self.setup_loaders()
        self.setup_optimizer_criterion()

    def setup_loaders(self) -> None:
        """
        Sets up the data loaders.
        """
        # Use config for paths, batch size, image size etc.
        full_dataset = ColorizationDataset(
            root_dir=self.config["data"]["train_dir"],
            target_size=tuple(self.config["data"]["image_size"]),
        )
        self.train_loader = DataLoader(
            full_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
        )
        # self.val_loader = DataLoader(...) # Setup validation loader

        print(f"Train loader setup with {len(full_dataset)} images.")

    def setup_optimizer_criterion(self) -> None:
        """
        Sets up the optimizer and criterion.
        """
        self.criterion = nn.MSELoss()  # Or other appropriate loss
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
            outputs = self.model(inputs)  # Uses the passed model (ResNet or ViT)
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
        # Implement evaluation logic using self.val_loader
        print("Evaluation not implemented yet.")
        pass

    def run_training(self) -> None:
        """
        Runs the full training loop.
        """
        num_epochs = self.config["training"]["num_epochs"]
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(1, num_epochs + 1):
            self.train_epoch(epoch)
            # Optional: self.evaluate()
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
