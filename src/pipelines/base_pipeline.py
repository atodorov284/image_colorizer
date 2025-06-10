from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import os


class BasePipeline(ABC):
    """
    Abstract base class for training pipelines.
    """

    def __init__(self, config: dict, model: nn.Module, device: torch.device) -> None:
        """
        Initialize the pipeline.

        Args:
            config (dict): Configuration dictionary.
            model (nn.Module): Model to be trained.
            device (torch.device): Device to use for training.
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Common setup
        self.checkpoint_dir = self.config["output"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_dir = self.config["output"]["best_model_dir"]
        os.makedirs(self.best_model_dir, exist_ok=True)

    @abstractmethod
    def setup_loaders(self) -> None:
        """Set up training and validation data loaders."""
        pass

    @abstractmethod
    def setup_optimizer_criterion(self) -> None:
        """Set up optimizer and loss function."""
        pass

    @abstractmethod
    def train_epoch(self, epoch_num: int) -> float:
        """Run one epoch of training.

        Args:
            epoch_num (int): The current epoch number.

        Returns:
            float: The average loss for the epoch.
        """
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the model on the validation set.

        Returns:
            float: The average loss on the validation set.
        """
        pass

    @abstractmethod
    def run_training(self) -> None:
        """Run the full training loop."""
        pass

    @abstractmethod
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Make a prediction on a single input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted tensor.
        """
        pass
