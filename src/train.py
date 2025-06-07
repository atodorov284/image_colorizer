import torch
import yaml

# Adjust imports based on final structure
from models.resnet import ResNetColorizationModel
from models.vgg import VGGColorizationModel
from models.vit import ViTColorizationModel
from pipelines.colorization_pipeline import ColorizationPipeline

if __name__ == "__main__":
    with open("src/configs/vgg_config.yaml", "r") as file:
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
    elif config["model"]["name"] == "vgg":
        model_instance = VGGColorizationModel(pretrained=config["model"]["pretrained"])
        print("Using VGG model.")
    else:
        raise ValueError(f"Unknown model type: {config['model']['name']}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model_instance = torch.nn.DataParallel(
            model_instance, device_ids=list(range(torch.cuda.device_count()))
        )

    print(model_instance)
    print(
        f"Number of parameters: {sum(p.numel() for p in model_instance.parameters())}"
    )

    pipeline = ColorizationPipeline(config, model_instance, device)

    pipeline.run_training()
