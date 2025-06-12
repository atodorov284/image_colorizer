import hashlib
import json
import os
from typing import Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.colorization_utils import ColorizationUtils
from utils.filtering_utils import FiltersUtils


class ColorizationDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing images for colorization.

    Args:
        root_dir (str): Path to the directory containing images.
        target_size (Tuple[int, int], optional): Target size (height, width)
                                                 for resizing images. Defaults to (256, 256).
    """

    def __init__(
        self,
        root_dir: str,
        captions_dir: str,
        target_size: Tuple[int, int] = (224, 224),
        cache_path: str = ".filtered_images.json",
    ) -> None:
        """Constructor for ColorizationDataset."""
        self.root_dir: str = root_dir
        self.target_size: Tuple[int, int] = target_size
        self.cache_path = cache_path

        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                cache = json.load(f)
            if cache.get("fingerprint") == self._fingerprint():
                self.image_files = cache["image_files"]
                print(f"Loaded {len(self.image_files)} image names from cache.")
                return
            else:
                print("Dataset changed, since last run. Filtering again.")
        else:
            print(f"Cache not found at {self.cache_path}. Filtering images.")

        self.image_files = self._build_filelist(captions_dir)

        with open(self.cache_path, "w") as f:
            json.dump(
                {"fingerprint": self._fingerprint(), "image_files": self.image_files}, f
            )
        print(f"Cache saved to {self.cache_path} for future runs.")

    def _build_filelist(self, captions_dir: str) -> list[str]:
        """Builds a list of image file names."""
        all_image_files: list[str] = [
            f
            for f in os.listdir(self.root_dir)
            if os.path.isfile(os.path.join(self.root_dir, f))
        ]
        coco = COCO(captions_dir)
        kept = []
        for name in tqdm(all_image_files, desc="Filtering images"):
            path = os.path.join(self.root_dir, name)
            if FiltersUtils.channel_difference_test(path):
                continue
            if FiltersUtils.caption_keywords_test(path, coco):
                continue
            kept.append(name)
        print(f"Keeping {len(kept)} / {len(all_image_files)} images.")
        print(f"Filtered {len(all_image_files) - len(kept)} images.")
        return kept

    def _fingerprint(self) -> str:
        """Returns a fingerprint of the dataset."""
        md5 = hashlib.md5()
        for name in sorted(
            f
            for f in os.listdir(self.root_dir)
            if os.path.isfile(os.path.join(self.root_dir, f))
        ):
            md5.update(name.encode())
        return md5.hexdigest()

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads, resizes, and preprocesses a single image from the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input (LLL) and Target (AB) tensors.
        """
        img_path: str = os.path.join(self.root_dir, self.image_files[idx])
        img_rgb_pil: Image.Image = Image.open(img_path).convert("RGB")
        input_tensor, output_tensor = ColorizationUtils.preprocess_image(
            img_rgb_pil, self.target_size
        )
        return input_tensor, output_tensor
