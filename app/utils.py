import base64
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageOps


def get_example_images(directory_path, limit=5):
    """
    Get a list of example image paths from the specified directory.

    Args:
        directory_path (str): Path to the directory containing images
        limit (int): Maximum number of example images to return

    Returns:
        list: List of paths to example images
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


def prepare_image_display(image, size=(256, 256)):
    """
    Prepare an image for display (resize, convert)

    Args:
        image (PIL.Image): Input image
        size (tuple): Target size

    Returns:
        PIL.Image: Processed image
    """
    # Make a copy of the image to avoid modifying the original
    img_copy = image.copy()

    # Resize
    img_copy = img_copy.resize(size)

    return img_copy


def create_comparison_figure(original, colorized, figsize=(10, 5)):
    """
    Create a matplotlib figure comparing original and colorized images

    Args:
        original (np.ndarray): Original image
        colorized (np.ndarray): Colorized image
        figsize (tuple): Figure size

    Returns:
        str: Base64 encoded figure
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


def ensure_directory_exists(path):
    """
    Ensure that a directory exists, creating it if necessary

    Args:
        path (str): Path to the directory
    """
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")


def save_uploaded_image(uploaded_file, save_dir="app/uploaded_images"):
    """
    Save an uploaded image file to disk

    Args:
        uploaded_file: Streamlit uploaded file
        save_dir (str): Directory to save the file

    Returns:
        str: Path to the saved file
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
