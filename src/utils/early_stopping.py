from torch import nn


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """Initialize the EarlyStopping object."""
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if early stopping should be triggered."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best model weights."""
        if self.best_state:
            model.load_state_dict(self.best_state)
