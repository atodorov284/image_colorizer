import torch
import yaml
import os
from PIL import Image
from torchvision import transforms

from models.resnet import ResNetColorizationModel
from models.vgg import VGGColorizationModel
from utils.predicting_utils import PredictingUtils
from utils.colorization_utils import ColorizationUtils

class ModelHub:
    """Load, cache and serve model instances for the API."""

    def __init__(self) -> None:
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._load_resnet()
        self._load_vgg()
        self._load_quantized_vgg()

    def _load_resnet(self) -> None:
        with open("configs/resnet_config.yaml", "r") as fh:
            cfg_res = yaml.safe_load(fh)
        self.res_target_size = tuple(cfg_res["data"]["image_size"])

        ckpt = f"../{cfg_res['output']['best_model_dir']}/best_model.pth"
        print(f"Loading ResNet model from {ckpt}")
        if os.path.exists(ckpt):
            model = ResNetColorizationModel(pretrained=False)
            state_dict = torch.load(ckpt, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.resnet = model
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Please train the model first.")

    def _load_vgg(self) -> None:
        with open("configs/vgg_config.yaml", "r") as fh:
            cfg_vgg = yaml.safe_load(fh)
        self.vgg_target_size = tuple(cfg_vgg["data"]["image_size"])

        ckpt = f"../{cfg_vgg['output']['best_model_dir']}/best_model.pth"
        print(f"Loading VGG model from {ckpt}")
        if os.path.exists(ckpt):
            model = VGGColorizationModel(pretrained=False)
            state_dict = torch.load(ckpt, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.vgg = model
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Please train the model first.")
        
    def _load_quantized_vgg(self) -> None:
        """Load the quantized VGG model."""
        with open("configs/vgg_config.yaml", "r") as fh:
            cfg_vgg = yaml.safe_load(fh)

        ckpt = f"../{cfg_vgg['output']['best_model_dir']}/best_model_dynamic_int8.pth"
        print(f"Loading VGG model from {ckpt}")
        if os.path.exists(ckpt):
            model = VGGColorizationModel(pretrained=False)
            model = torch.load(ckpt, map_location=self.device, weights_only=False)
            model.to(self.device).eval()
            self.quant = model
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Please train the model first.")

    def colorize_with_resnet(self, pil_rgb: Image.Image) -> Image.Image:
        """Colorise *pil_rgb* using the ResNet colourisation model."""
        with torch.no_grad():
            rgb_np = PredictingUtils.predict_resnet(self.resnet, self.device, pil_rgb, pil_rgb.size)

        return Image.fromarray((rgb_np * 255).clip(0, 255).astype("uint8"))

    def colorize_with_vgg(self, pil_rgb: Image.Image) -> Image.Image:
        """Colorise *pil_rgb* using the VGG colourisation model."""
        with torch.no_grad():
            rgb_np = PredictingUtils.predict_vgg(self.vgg, self.device, pil_rgb, pil_rgb.size)

        return Image.fromarray((rgb_np * 255).clip(0, 255).astype("uint8"))
    
    def colorize_with_quantized_vgg(self, pil_rgb: Image.Image) -> Image.Image:
        """Colorise *pil_rgb* using the quantized VGG colourisation model."""
        with torch.no_grad():
            rgb_np = PredictingUtils.predict_vgg(self.quant, self.device, pil_rgb, pil_rgb.size)

        return Image.fromarray((rgb_np * 255).clip(0, 255).astype("uint8"))
