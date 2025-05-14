import io
import os
import sys
from pathlib import Path

# Add the project root to the path before other imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
src_path = os.path.join(Path(__file__).parent.parent, "src")
sys.path.insert(0, src_path)

# Standard imports
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

# Import our app modules
from app.model_loader import get_available_models, load_model
from app.utils import get_example_images, save_uploaded_image
from src.utils.colorization_utils import ColorizationUtils

# Page configuration
st.set_page_config(
    page_title="Image Colorization App",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    "<h1 class='main-header'>🎨 Image Colorization App</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p class='info-text'>Transform grayscale images into vibrant colored versions using deep learning.</p>",
    unsafe_allow_html=True,
)

# Sidebar for model selection and settings
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Model Settings</h2>", unsafe_allow_html=True)

    # Get available models
    available_models = get_available_models()
    model_names = [model["name"] for model in available_models]

    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        model_names,
        index=len(model_names) - 1 if "Dummy Colorizer" in model_names else 0,
    )

    # Find the selected model info
    selected_model = next(
        (model for model in available_models if model["name"] == model_choice), None
    )

    # Display model info
    if selected_model:
        if selected_model.get("is_dummy", False):
            st.info("⚠️ Using demo mode with random colorization")
        elif selected_model["is_trained"]:
            st.success(f"✅ {selected_model['name']} (Trained)")
        else:
            st.warning(f"⚠️ {selected_model['name']} (Untrained)")

        if "description" in selected_model:
            st.info(selected_model["description"])

    # Image Settings
    st.markdown("<h2 class='sub-header'>Image Settings</h2>", unsafe_allow_html=True)
    img_size = st.slider("Image Size", min_value=128, max_value=512, value=256, step=64)

    # Example images
    st.markdown("<h2 class='sub-header'>Example Images</h2>", unsafe_allow_html=True)

    # Get example images
    test_img_dir = Path("data/test2017")
    example_images = get_example_images(test_img_dir, limit=5)
    if example_images:
        default_img_path = str(example_images[0])
        st.info(f"Found {len(example_images)} example images")
    else:
        default_img_path = None
        st.warning("No example images found.")


# Helper function to preprocess image in fallback mode
def preprocess_image_fallback(image, target_size):
    """Simple preprocessing when ColorizeUtils is not available"""
    # Resize the image
    img_resized = image.resize(target_size)

    # Convert to grayscale - this is for display
    gray = img_resized.convert("L")

    # Create normalized input - just for compatibility
    l_channel = np.array(gray) / 255.0

    # Stack to create 3-channel input (simulate LLL tensor)
    input_tensor = torch.from_numpy(np.stack([l_channel] * 3, axis=0)).float()

    return input_tensor, None


# Helper function to reconstruct image in fallback mode
def reconstruct_image_fallback(input_tensor, ab_tensor):
    """Simple image reconstruction when ColorizeUtils is not available"""
    # Get dimensions
    if hasattr(input_tensor, "shape"):
        _, h, w = input_tensor.shape
    else:
        h, w = 256, 256

    # If we have a proper tensor, convert to numpy
    if hasattr(input_tensor, "cpu") and hasattr(input_tensor, "numpy"):
        l_np = input_tensor[0].cpu().numpy()
    elif hasattr(input_tensor, "numpy"):
        l_np = input_tensor[0].numpy()
    else:
        l_np = input_tensor[0]

    # Create a random colorized image
    # This is a simple HSV colorization where we use the grayscale as V
    # and generate random H and constant S
    h_channel = np.random.random((h, w)) if ab_tensor is None else ab_tensor[0]
    s_channel = np.ones((h, w)) * 0.6 if ab_tensor is None else (ab_tensor[1] + 1) / 2
    v_channel = l_np

    # Stack to create HSV
    hsv = np.stack([h_channel, s_channel, v_channel], axis=2)

    # Convert HSV to RGB (approximation)
    # This is a simple conversion that doesn't require scikit-image
    hi = np.floor(hsv[..., 0] * 6)
    f = hsv[..., 0] * 6 - hi
    p = hsv[..., 2] * (1 - hsv[..., 1])
    q = hsv[..., 2] * (1 - f * hsv[..., 1])
    t = hsv[..., 2] * (1 - (1 - f) * hsv[..., 1])

    hi = hi.astype(np.int32) % 6
    rgb = np.zeros_like(hsv)

    rgb[hi == 0, 0] = hsv[hi == 0, 2]
    rgb[hi == 0, 1] = t[hi == 0]
    rgb[hi == 0, 2] = p[hi == 0]

    rgb[hi == 1, 0] = q[hi == 1]
    rgb[hi == 1, 1] = hsv[hi == 1, 2]
    rgb[hi == 1, 2] = p[hi == 1]

    rgb[hi == 2, 0] = p[hi == 2]
    rgb[hi == 2, 1] = hsv[hi == 2, 2]
    rgb[hi == 2, 2] = t[hi == 2]

    rgb[hi == 3, 0] = p[hi == 3]
    rgb[hi == 3, 1] = q[hi == 3]
    rgb[hi == 3, 2] = hsv[hi == 3, 2]

    rgb[hi == 4, 0] = t[hi == 4]
    rgb[hi == 4, 1] = p[hi == 4]
    rgb[hi == 4, 2] = hsv[hi == 4, 2]

    rgb[hi == 5, 0] = hsv[hi == 5, 2]
    rgb[hi == 5, 1] = p[hi == 5]
    rgb[hi == 5, 2] = q[hi == 5]

    return rgb


# Function to load model
@st.cache_resource
def get_pipeline(model_name, img_size):
    """Load the colorization model and create a pipeline."""
    # Get the selected model info
    selected_model = next(
        (model for model in available_models if model["name"] == model_name), None
    )

    if not selected_model:
        st.error(f"Model {model_name} not found!")
        return None, False

    # Create a simple config for the pipeline
    config = {
        "data": {"image_size": [img_size, img_size]},
        "output": {
            "checkpoint_dir": "src/models/checkpoints",
            "best_model_dir": "src/models/checkpoints",
        },
        "training": {
            "batch_size": 1,
            "learning_rate": 0.001,
            "num_epochs": 10,
            "patience": 5,
            "min_delta": 0.001,
        },
    }

    # Load the model and pipeline
    pipeline, is_trained = load_model(selected_model, config)
    return pipeline, is_trained


# Load the model
with st.spinner("Loading model..."):
    pipeline, is_trained = get_pipeline(model_choice, img_size)
    if pipeline:
        model_type = (
            "(Demo)"
            if hasattr(pipeline, "name") and pipeline.name == "Dummy Colorizer"
            else "(Trained)"
            if is_trained
            else "(Untrained)"
        )
        st.success(f"Model loaded successfully! {model_type}")
    else:
        st.error("Failed to load model. Using fallback mode.")


# Main content
st.markdown(
    "<h2 class='sub-header'>Upload or Select an Image</h2>", unsafe_allow_html=True
)

# Image selection
upload_col, sample_col = st.columns(2)

with upload_col:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

with sample_col:
    use_example = st.button("Use Example Image", type="primary")

# Image processing
if uploaded_file is not None:
    image_path = uploaded_file
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.image = image
elif use_example and default_img_path:
    image_path = default_img_path
    image = Image.open(default_img_path).convert("RGB")
    st.session_state.image = image
else:
    image = st.session_state.get("image", None)

# Display results
if image is not None:
    # Resize the image
    image = image.resize((img_size, img_size))

    # Convert to grayscale for display
    grayscale_image = image.convert("L").convert("RGB")

    # Process the image
    with st.spinner("Colorizing image..."):
        # Prepare input for the model
        try:
            input_tensor, _ = ColorizationUtils.preprocess_image(
                image, (img_size, img_size)
            )
        except Exception:
            input_tensor, _ = preprocess_image_fallback(image, (img_size, img_size))

        # Make prediction
        try:
            with torch.no_grad():
                predicted_ab = pipeline.predict(input_tensor)

            # Reconstruct the image
            try:
                colorized_image_np = ColorizationUtils.reconstruct_image(
                    input_tensor, predicted_ab
                )
            except Exception:
                colorized_image_np = reconstruct_image_fallback(
                    input_tensor, predicted_ab
                )

            colorized_image = Image.fromarray(
                (colorized_image_np * 255).astype(np.uint8)
            )
        except Exception as e:
            st.error(f"Error during colorization: {str(e)}")
            st.info("Using fallback random colorization")

            # Fallback to random colorization
            colorized_image_np = reconstruct_image_fallback(input_tensor, None)
            colorized_image = Image.fromarray(
                (colorized_image_np * 255).astype(np.uint8)
            )

    # Display images
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "<p style='text-align:center'>Original Image</p>", unsafe_allow_html=True
        )
        st.image(image, use_column_width=True)

    with col2:
        st.markdown(
            "<p style='text-align:center'>Grayscale Input</p>", unsafe_allow_html=True
        )
        st.image(grayscale_image, use_column_width=True)

    with col3:
        st.markdown(
            "<p style='text-align:center'>Colorized Result</p>", unsafe_allow_html=True
        )
        st.image(colorized_image, use_column_width=True)

    # Download buttons
    col1, col2 = st.columns(2)

    with col1:
        # Save image to bytes
        img_byte_arr = io.BytesIO()
        colorized_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        st.download_button(
            label="Download Colorized Image",
            data=img_byte_arr,
            file_name="colorized_image.png",
            mime="image/png",
        )

    with col2:
        # Show comparison slider
        st.markdown(
            "<p style='text-align:center'>Original vs. Colorized Comparison</p>",
            unsafe_allow_html=True,
        )
        st.image([image, colorized_image], width=img_size)

# Add information about the model and project
st.markdown("---")
st.markdown("<h2 class='sub-header'>About This Project</h2>", unsafe_allow_html=True)
st.markdown("""
This app demonstrates an image colorization model that can convert grayscale images to color.
The model uses deep learning to predict the color channels of an image based on its grayscale information.

**How it works:**
1. The input image is converted to the LAB color space
2. The L channel (lightness) is used as input to the model
3. The model predicts the A and B channels (color information)
4. The predicted A and B channels are combined with the original L channel
5. The resulting LAB image is converted back to RGB

**Technologies used:**
- PyTorch for deep learning
- ResNet architecture for image processing
- Streamlit for the web interface
""")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center'>🎨 Image Colorization Project</p>",
    unsafe_allow_html=True,
)
