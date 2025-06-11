import torch
import yaml

from models.resnet import ResNetColorizationModel
from models.vgg import VGGColorizationModel
from pipelines.colorization_pipeline import ColorizationPipeline


def main():
    # Load config
    with open("src/configs/vgg_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Setup device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Create model based on config
    model_name = config["model"]["name"].lower()
    if model_name == "resnet":
        model_instance = ResNetColorizationModel(
            pretrained=config["model"]["pretrained"]
        )
        print("Using ResNet model.")
    elif model_name == "vgg":
        model_instance = VGGColorizationModel(pretrained=config["model"]["pretrained"])
        print("Using VGG model.")
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model_instance = torch.nn.DataParallel(
            model_instance, device_ids=list(range(torch.cuda.device_count()))
        )

    print(f"Model parameters: {sum(p.numel() for p in model_instance.parameters()):,}")

    # Create unified pipeline
    pipeline = ColorizationPipeline(config, model_instance, device)

    # Run training
    pipeline.run_training()


if __name__ == "__main__":
    main()
