import base64
import io
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageOps


def get_example_images(directory_path: Union[str, Path], limit: int = 5) -> List[Path]:
    """
    Get a list of example image paths from the specified directory.

    This function scans a directory for image files with common extensions and returns
    a list of paths to these images, limited by the specified count.

    Args:
        directory_path: Path to the directory containing images
        limit: Maximum number of example images to return

    Returns:
        List[Path]: List of paths to example images, sorted alphabetically and limited
                   to the specified count. Returns empty list if directory doesn't exist.
    """
    directory = Path(directory_path)
    if not directory.exists():
        return []

    image_extensions = [".jpg", ".jpeg", ".png"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(list(directory.glob(f"*{ext}")))

    # Take only the first 'limit' images
    return sorted(image_paths)[:limit]


def prepare_image_display(
    image: Image.Image, size: Tuple[int, int] = (256, 256)
) -> Image.Image:
    """
    Prepare an image for display by resizing it to the target dimensions.

    Creates a copy of the input image and resizes it to avoid modifying the original.

    Args:
        image: Input PIL Image object
        size: Target size as (width, height) tuple

    Returns:
        Image.Image: Processed image ready for display
    """
    # Make a copy of the image to avoid modifying the original
    img_copy = image.copy()

    # Resize
    img_copy = img_copy.resize(size)

    return img_copy


def create_comparison_figure(
    original: np.ndarray, colorized: np.ndarray, figsize: Tuple[int, int] = (10, 5)
) -> Optional[str]:
    """
    Create a matplotlib figure comparing original and colorized images.

    This function creates a side-by-side comparison of the original and
    colorized images, encodes it as a base64 string suitable for displaying
    in HTML/web contexts.

    Args:
        original: Original image as numpy array
        colorized: Colorized image as numpy array
        figsize: Figure size as (width, height) tuple in inches

    Returns:
        Optional[str]: Base64 encoded figure as a data URL string that can be used in HTML,
                      or None if an error occurs during figure creation
    """
    try:
        fig = Figure(figsize=figsize)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(original)
        ax1.set_title("Original")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(colorized)
        ax2.set_title("Colorized")
        ax2.axis("off")

        # Save figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        # Encode buffer to base64
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    except Exception as e:
        print(f"Error creating comparison figure: {e}")
        return None


def ensure_directory_exists(path: Union[str, Path]) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    This function attempts to create the specified directory and all parent
    directories if they don't already exist.

    Args:
        path: Path to the directory to be created
    """
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")


def save_uploaded_image(
    uploaded_file: Any, save_dir: str = "app/uploaded_images"
) -> Optional[str]:
    """
    Save an uploaded image file to disk.

    This function saves a file uploaded through Streamlit to the specified
    directory, creating the directory if it doesn't exist.

    Args:
        uploaded_file: Streamlit UploadedFile object containing the image data
        save_dir: Directory path where the file should be saved

    Returns:
        Optional[str]: Path to the saved file, or None if an error occurs
    """
    try:
        ensure_directory_exists(save_dir)
        file_path = os.path.join(save_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path
    except Exception as e:
        print(f"Error saving uploaded image: {e}")
        return None
