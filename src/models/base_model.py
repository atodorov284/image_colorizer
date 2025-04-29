from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseColorizationModel(nn.Module, ABC):
    """
    Abstract Base Class for Colorization Models.

    This base class enforces the implementation of the forward() method.
    """

    def __init__(self) -> None:
        """
        Constructor for the BaseColorizationModel class.
        """
        super().__init__()

    @abstractmethod
    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the colorization model.

        Args:
            x_l (torch.Tensor): Input tensor (e.g., L channel or LLL).
                                Shape typically [Batch, Channels, H, W].

        Returns:
            torch.Tensor: Predicted AB channels tensor. Shape [Batch, 2, H, W].
        """
        pass
