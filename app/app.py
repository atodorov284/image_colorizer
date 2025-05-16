import io
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple, Union

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
import io
import base64
import streamlit as st
import streamlit.components.v1 as components

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

def create_comparison_slider(original_img, colorized_img, height=400):
    """Create a slider for comparing before/after images using streamlit components.
    
    Args:
        original_img: Original PIL image (grayscale)
        colorized_img: Colorized PIL image
        height: Height of the comparison widget in pixels
        
    Returns:
        None - directly renders the component
    """
    import base64
    import io
    
    # Ensure images are the same size, original_img is used as reference
    width, _ = original_img.size 
    # If colorized_img has a different size, resize it to match original_img.
    # This assumes original_img and colorized_img should be aligned.
    if colorized_img.size != original_img.size:
        colorized_img = colorized_img.resize(original_img.size)

    orig_bytes = io.BytesIO()
    colorized_bytes = io.BytesIO()
    original_img.save(orig_bytes, format="PNG")
    colorized_img.save(colorized_bytes, format="PNG")
    
    orig_base64 = base64.b64encode(orig_bytes.getvalue()).decode()
    colorized_base64 = base64.b64encode(colorized_bytes.getvalue()).decode()
    
    # Unique ID for elements to avoid conflicts if multiple sliders are on a page
    component_id = f"comparison-slider-{int(np.random.rand() * 1000000)}"

    html_code = f"""
    <div id="{component_id}" style="position: relative; width: 100%; height: {height}px; overflow: hidden;">
        <img src="data:image/png;base64,{orig_base64}" 
             style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;" 
             alt="Original Image" />
        <div class="colorized-overlay" 
             style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden;">
            <img src="data:image/png;base64,{colorized_base64}" 
                 style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; clip-path: inset(0 100% 0 0);" 
                 alt="Colorized Image" />
        </div>
        <input type="range" class="slider" min="0" max="100" value="0" 
               style="position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); width: 90%; z-index: 10;" 
               aria-label="Comparison Slider"/>
        <div class="divider" 
             style="position: absolute; top: 0; bottom: 0; left: 0%; width: 3px; background: rgba(255, 255, 255, 0.7); box-shadow: 0 0 3px rgba(0,0,0,0.5); z-index: 5; cursor: ew-resize;">
        </div>
    </div>

    <script>
    (function() {{
        // Ensure this script runs for the correct component instance
        const currentDoc = document.currentScript.ownerDocument;
        const container = currentDoc.getElementById("{component_id}");
        if (!container) {{
            console.error("Slider container not found: {component_id}");
            return;
        }}
        
        const slider = container.querySelector('.slider');
        const colorizedOverlay = container.querySelector('.colorized-overlay img'); // Target the img inside the overlay
        const divider = container.querySelector('.divider');
        
        function updateVisuals(percent) {{
            // The clip-path reveals 'percent' of the image from the left.
            // inset(top right bottom left)
            // To show 'percent' from left, we clip (100-percent) from the right.
            const clipValue = 100 - parseFloat(percent);
            colorizedOverlay.style.clipPath = `inset(0 ${{clipValue}}% 0 0)`;
            divider.style.left = percent + '%';
        }}
        
        // Initial setup based on slider's current value (which is 0)
        updateVisuals(slider.value); 
        
        slider.addEventListener('input', function(e) {{
            updateVisuals(e.target.value);
        }});

        // Optional: Make the divider draggable (basic implementation)
        let isDragging = false;
        divider.addEventListener('mousedown', function(e) {{
            isDragging = true;
            container.style.cursor = 'ew-resize';
        }});
        container.addEventListener('mouseup', function(e) {{
            if (isDragging) {{
                isDragging = false;
                container.style.cursor = 'default';
            }}
        }});
        container.addEventListener('mouseleave', function(e) {{ // Stop dragging if mouse leaves container
             if (isDragging) {{
                isDragging = false;
                container.style.cursor = 'default';
            }}
        }});
        container.addEventListener('mousemove', function(e) {{
            if (isDragging) {{
                const rect = container.getBoundingClientRect();
                let percent = ((e.clientX - rect.left) / rect.width) * 100;
                percent = Math.max(0, Math.min(100, percent)); // Clamp between 0 and 100
                slider.value = percent;
                updateVisuals(percent);
            }}
        }});
    }})();
    </script>
    """
    components.html(html_code, height=height + 70)
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
    .image-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 1rem;
    }
    .image-title {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #166ABF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .stDownloadButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    .stDownloadButton>button:hover {
        background-color: #3E8E41;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .stFileUploader>div>div {
        display: flex;
        justify-content: center;
    }
    div[data-testid="stFileUploader"] div:first-child {
        width: 100%;
        justify-content: center;
    }
    div[data-testid="stFileUploader"] div button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    /* Hide the X button and file icon in the file uploader */
    .st-emotion-cache-1pbsqtx {
        display: none !important;
    }
    /* Hide any SVG with the file path in the uploader */
    svg path[d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6z"] {
        display: none !important;
    }
    /* Additional selector for the button that contains the X icon */
    div[data-testid="stFileUploaderRemoveFile"] {
        display: none !important;
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
    "<p class='info-text' style='text-align: center;'>Transform grayscale images into vibrant colored versions using deep learning.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h2 class='sub-header' style='text-align: center;'>About This Project</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div style="max-width: 800px; margin: 0 auto; text-align: center;">
    <p>This app demonstrates an image colorization model that can convert grayscale images to color.
    The model uses deep learning to predict the color channels of an image based on its grayscale information.</p>
</div>
""",
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
    img_size = 256

    # Get example images
    test_img_dir = Path("data/test2017")
    example_images = get_example_images(test_img_dir, limit=5)
    if example_images:
        default_img_path = str(example_images[0])
    else:
        default_img_path = None


# Helper function to preprocess image in fallback mode
def preprocess_image_fallback(
    image: Image.Image, target_size: Tuple[int, int]
) -> Tuple[torch.Tensor, None]:
    """
    Simple image preprocessing when ColorizeUtils is not available.

    This is a fallback function that converts an image to grayscale and normalizes it
    for input to the colorization model.

    Args:
        image: Input PIL image
        target_size: Target size (width, height) to resize the image to

    Returns:
        Tuple[torch.Tensor, None]:
            - A tensor of shape (3, H, W) with the grayscale channel replicated
            - None as a placeholder for the second return value expected by the main function
    """
    # Resize the image
    img_resized = image.resize(target_size)

    # Convert to grayscale - this is for display
    gray = img_resized.convert("L")

    # Create normalized input - just for compatibility
    l_channel = np.array(gray) / 255.0

    # Stack to create 3-channel input (simulate LLL tensor)
    input_tensor = torch.from_numpy(np.stack([l_channel] * 3, axis=0)).float()

    return input_tensor, None


# Function to load model
@st.cache_resource
def get_pipeline(model_name: str, img_size: int) -> Tuple[Union[Any, None], bool]:
    """
    Load the colorization model and create a pipeline.

    This function loads the specified model and creates a pipeline for image colorization.
    The result is cached by Streamlit to avoid reloading on each rerun.

    Args:
        model_name: Name of the model to load
        img_size: Size of images to be processed (square images, width=height=img_size)

    Returns:
        Tuple[Union[Any, None], bool]:
            - The colorization pipeline object (or None if loading failed)
            - Boolean indicating whether the model is trained
    """
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


# Main content
st.markdown(
    "<h2 class='sub-header' style='text-align: center;'>Upload or Select an Image</h2>",
    unsafe_allow_html=True,
)

# File uploader in a centered column
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )


if "image" not in st.session_state:
    st.session_state.image = None

# Image processing
if uploaded_file is not None:
    image_path = uploaded_file
    image = Image.open(uploaded_file).convert("RGB")
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
        input_tensor, _ = ColorizationUtils.preprocess_image(
            image, (img_size, img_size)
        )
        with torch.no_grad():
            predicted_ab = pipeline.predict(input_tensor)

        colorized_image_np = ColorizationUtils.reconstruct_image(
            input_tensor, predicted_ab
        )

        colorized_image = Image.fromarray((colorized_image_np * 255).astype(np.uint8))
    
    # Directly call the function to render the slider
    st.markdown("<h3>Drag to Compare Before/After</h3>", unsafe_allow_html=True)
    create_comparison_slider(grayscale_image, colorized_image) # This will now render the slider

    # Display images in a row with titles
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.markdown("<p class='image-title'>Original Image</p>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.markdown(
            "<p class='image-title'>Grayscale Image</p>", unsafe_allow_html=True
        )
        st.image(grayscale_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.markdown(
            "<p class='image-title'>Colorized Result</p>", unsafe_allow_html=True
        )
        st.image(colorized_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Download button - centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
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


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center'>🎨 Image Colorization Project</p>",
    unsafe_allow_html=True,
)
