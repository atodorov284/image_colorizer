from PIL import Image
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO

class FiltersUtils:
    """Utility class for image filtering functions."""

    TOLERANCE = 30

    @staticmethod
    def channel_difference_test(img_path: str, tol: int = TOLERANCE):
        """
        Checks if the RGB channels of an image are within a specified tolerance.
        Args:
            img_path (str): Path to the image file.
            tol (int): Tolerance level for channel difference.
        Returns:
            bool: True if the image is grayscale.
        """
        img = Image.open(img_path)
        rgb = np.asarray(img.convert("RGB"), dtype=np.int16)  # (H,W,3)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        max_diff = np.maximum(
            np.abs(r - g),
            np.maximum(np.abs(r - b), np.abs(g - b))
        )
        return np.all(max_diff <= tol)
    

    keywords = []

    # numeric decades / dates ----------------------------------------------------
    a1_list = [f"{i}s"          for i in range(1900, 2000)]         # "1900s", …
    a2_list = [f"{i}"           for i in range(1900, 2000)]         # "1900", …
    a3_list = [f"year {i}"      for i in range(1900, 2000)]         # "year 1900", …
    a4_list = [f"circa {i}"     for i in range(1900, 2000)]         # "circa 1900", …

    b1_list = [f"{y[0]} {y[1]} {y[2]} {y[3]} s"           for y in a1_list]  # "1 9 0 0 s"
    b2_list = [f"{y[0]} {y[1]} {y[2]} {y[3]}"             for y in a1_list]  # "1 9 0 0"
    b3_list = [f"year {y[0]} {y[1]} {y[2]} {y[3]}"        for y in a1_list]
    b4_list = [f"circa {y[0]} {y[1]} {y[2]} {y[3]}"       for y in a1_list]

    # hard‑coded phrases ---------------------------------------------------------
    words_list = [
        "black and white,", "black and white", "black & white,", "black & white",
        "circa",
        "balck and white,",
        "monochrome,", "black-and-white,", "black-and-white photography,",
        "black - and - white photography,", "monochrome bw,",
        "black white,", "black an white,",
        "grainy footage,", "grainy footage", "grainy photo,", "grainy photo",
        "b&w photo", "back and white", "back and white,", "monochrome contrast",
        "monochrome", "grainy", "grainy photograph,", "grainy photograph",
        "low contrast,", "low contrast", "b & w", "grainy black-and-white photo,",
        "bw,", "grainy black-and-white photo", "b & w,", "b&w,",
        "b&w!,", "b&w", "black - and - white,", "bw photo,", "grainy  photo,",
        "black-and-white photo,", "black-and-white photo",
        "black - and - white photography",
        "b&w photo,", "monochromatic photo,", "grainy monochrome photo,",
        "monochromatic", "blurry photo,", "blurry,", "blurry photography,",
        "monochromatic photo", "black - and - white photograph,",
        "black - and - white photograph", "black on white,", "black on white",
        "black-and-white", "historical image,", "historical picture,",
        "historical photo,", "historical photograph,", "archival photo,",
        "taken in the early", "taken in the late",
        "historic photograph,", "restored,", "restored", "historical photo",
        "historical setting,", "historic photo,", "historic",
        "desaturated!!,", "desaturated!,", "desaturated,", "desaturated",
        "shot on leica", "shot on leica sl2", "sl2",
        "taken with a leica camera", "leica sl2", "leica",
    ]

    # combine --------------------------------------------------------------------
    keywords.extend(a1_list + a2_list + a3_list + a4_list)
    keywords.extend(b1_list + b2_list + b3_list + b4_list)
    keywords.extend(words_list)

    keywords = [w.lower() for w in keywords]

    @staticmethod
    def caption_keywords_test(img_path: str, coco: COCO, keywords: list = keywords):
        """
        Checks if the image caption contains any of the specified keywords.
        Args:
            img_path (str): Path to the image file.
            keywords (list): List of keywords to check against.
        Returns:
            bool: True if the image is potentially grayscale.
        """
        def get_captions(img_path: str):
            """
            Extracts captions from the COCO dataset for a given image path.
            Args:
                img_path (str): Path to the image file.
            Returns:
                List[str]: List of strings representing captions for the image.
            """
            img_id = int(Path(img_path).stem.split('_')[-1])     # e.g. train2017/000000391895.jpg → 391895
            anns = coco.imgToAnns[img_id]
            return [a['caption'] for a in anns]


        captions = get_captions(img_path)

        return any(any(k in c.lower() for k in keywords) for c in captions)