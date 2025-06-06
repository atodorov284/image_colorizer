import torch
import yaml

# Adjust imports based on final structure
from models.resnet import ResNetColorizationModel
from models.vit import ViTColorizationModel
from pipelines.colorization_pipeline import ColorizationPipeline

if __name__ == "__main__":
    with open("src/configs/vit_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if config["model"]["name"] == "resnet":
        model_instance = ResNetColorizationModel(
            pretrained=config["model"]["pretrained"]
        )
        print("Using ResNet model.")
    elif config["model"]["name"] == "vit":
        # model_instance = ViTColorizationModel()
        print("Using ViT model.")
        model_instance = ViTColorizationModel(
            pretrained=config["model"]["pretrained"],
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['name']}")

    pipeline = ColorizationPipeline(config, model_instance, device)

    pipeline.run_training()
