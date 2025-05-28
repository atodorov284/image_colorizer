import torch
import yaml
import os
from PIL import Image

from models.resnet import ResNetColorizationModel
from utils.colorization_utils import ColorizationUtils
from utils.predicting_utils import PredictingUtils

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

    def colorize_with_resnet(self, pil_rgb: Image.Image) -> Image.Image:  # noqa: D401
        """Colorise *pil_rgb* using the ResNet colourisation model."""
        lll, _ = ColorizationUtils.preprocess_image(pil_rgb, self.res_target_size)
        with torch.no_grad():
            ab = PredictingUtils.predict(self.resnet, self.device, lll)

        rgb_np = ColorizationUtils.reconstruct_image(lll, ab)
        return Image.fromarray((rgb_np * 255).clip(0, 255).astype("uint8"))
