from typing import Dict, Any, Union
import torch
import torch.nn as nn

from pipelines.resnet_pipeline import ResNetPipeline
from pipelines.vgg_pipeline import VGGPipeline


class ColorizationPipeline:
    """Unified pipeline factory for different colorization models."""
    
    PIPELINE_REGISTRY = {
        "resnet": ResNetPipeline,
        "vgg": VGGPipeline,
        # Can easily add more: "vit": ViTPipeline, "unet": UNetPipeline, etc.
    }
    
    def __init__(self, config: Dict[str, Any], model: nn.Module, device: torch.device):
        """
        Initialize the appropriate pipeline based on config.
        
        Args:
            config: Configuration dictionary
            model: The model to train
            device: Device to use for training
        """
        self.config = config
        self.model = model
        self.device = device
        
        # Determine pipeline type from config
        model_name = config["model"]["name"].lower()
        
        if model_name not in self.PIPELINE_REGISTRY:
            raise ValueError(f"Unknown model type: {model_name}. "
                           f"Available: {list(self.PIPELINE_REGISTRY.keys())}")
        
        # Create the appropriate pipeline
        pipeline_class = self.PIPELINE_REGISTRY[model_name]
        self.pipeline = pipeline_class(config, model, device)
        
        print(f"Initialized {pipeline_class.__name__} for model: {model_name}")
    
    def run_training(self) -> None:
        """Run training using the selected pipeline."""
        self.pipeline.run_training()
    
    def evaluate(self, **kwargs) -> float:
        """Evaluate using the selected pipeline."""
        return self.pipeline.evaluate(**kwargs)
    
    def predict(self, *args, **kwargs):
        """Make predictions using the selected pipeline."""
        return self.pipeline.predict(*args, **kwargs)
    
    def get_pipeline(self) -> Union[ResNetPipeline, VGGPipeline]:
        """Get the underlying pipeline instance."""
        return self.pipeline
    
    @classmethod
    def register_pipeline(cls, name: str, pipeline_class):
        """Register a new pipeline type."""
        cls.PIPELINE_REGISTRY[name] = pipeline_class
        print(f"Registered new pipeline: {name}")
