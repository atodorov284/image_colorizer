import io
from enum import Enum
from typing import Literal

import PIL
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from PIL import Image, ImageOps

from .model_hub import ModelHub


class ModelType(str, Enum):
    RESNET = "resnet"
    VIT = "vit"
    QUANT = "quant"


MODEL_HUB = ModelHub()

app = FastAPI(
    title="Grayscale → Colour API",
    summary="Upload a grayscale image, receive a colourised PNG using deep learning models.",
    description="""
# Grayscale to Color Image Conversion API

Transform your black and white images into vibrant, colorized versions using state-of-the-art deep learning models.

## Supported Image Formats
This API accepts all common image formats supported by PIL/Pillow:
- **JPEG** (.jpg, .jpeg)
- **PNG** (.png) 
- **TIFF** (.tiff, .tif)
- **BMP** (.bmp)
- **GIF** (.gif) - static images only
- **WebP** (.webp)
- And many others supported by PIL

## Available Models

### ResNet-18 Model (`/predict/resnet`)
- **Status**: ✅ Available
- **Architecture**: ResNet-18 based colorization network
- **Training**: Optimized for 256×256 pixel images
- **Performance**: Fast inference, good quality results

### Vision Transformer Model (`/predict/vit`) 
- **Status**: 🚧 Coming Soon
- **Architecture**: Vision Transformer based approach
- **Note**: Model training in progress

## Usage Examples

### cURL Example
```bash
curl -X POST "http://localhost:8000/predict/resnet" \
     -H "accept: image/png" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@path/to/your/grayscale_image.jpg" \
     --output colorized_result.png
```

### Python with requests
```python
import requests

url = "http://localhost:8000/predict/resnet"
files = {"image": open("grayscale_photo.jpg", "rb")}

response = requests.post(url, files=files)
if response.status_code == 200:
    with open("colorized_output.png", "wb") as f:
        f.write(response.content)
    print("Image colorized successfully!")
```

### JavaScript/Fetch API
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('/predict/resnet', {
    method: 'POST',
    body: formData
})
.then(response => response.blob())
.then(blob => {
    const imageUrl = URL.createObjectURL(blob);
    document.getElementById('result').src = imageUrl;
});
```

## Technical Details

### Image Processing Pipeline
1. **Input**: Uploaded image in any supported format
2. **Preprocessing**: 
   - EXIF orientation correction applied automatically
   - Conversion to RGB color space
   - Resizing to model's expected input size (256×256 for ResNet)
3. **Colorization**: Deep learning model predicts color channels
4. **Output**: High-quality PNG image stream

### Performance Notes
- Images are automatically resized to 256×256 pixels for optimal model performance
- Original aspect ratios are preserved during processing
- EXIF metadata is respected for proper image orientation

### Error Handling
- **415 Unsupported Media Type**: Invalid or corrupted image file
- **501 Not Implemented**: Model not available (ViT endpoint)
- **422 Validation Error**: Missing or invalid form data

## Response Format
All successful requests return a PNG image stream with:
- **Content-Type**: `image/png`
- **Status**: 200 OK
- **Body**: Binary PNG image data
""",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/", description="Root endpoint that redirects to API documentation.")
async def root():
    return RedirectResponse(url="/docs")


@app.post(
    "/predict",
    summary="Colorize image with selected model",
    description="""
    Upload a grayscale image and specify the model to use for colorization.
    
    **Models available**:
    - `resnet`: ResNet-18 based colorization (available)
    - `vit`: Vision Transformer based colorization (coming soon)
    
    **Supported formats**: JPEG, PNG, TIFF, BMP, WebP, GIF (static), and other PIL-compatible formats.
    
    **Processing**: Images are automatically resized to 256×256 pixels and EXIF orientation is corrected.
    
    **Output**: High-quality PNG image with predicted colors applied to the original grayscale input.
    """,
    responses={
        200: {
            "description": "Successfully colorized image",
            "content": {"image/png": {"example": "Binary PNG image data"}},
        },
        415: {
            "description": "Unsupported image format or corrupted file",
            "content": {
                "application/json": {"example": {"detail": "Invalid image file."}}
            },
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {"example": {"detail": "Invalid parameters"}}
            },
        },
        501: {
            "description": "Model not yet available",
            "content": {
                "application/json": {
                    "example": {"detail": "Selected model not available yet"}
                }
            },
        },
    },
)
async def predict(model: ModelType, image: UploadFile):
    """
    Colorize a grayscale image using the specified model.

    Args:
        model: Model to use for colorization ("resnet" or "vit")
        image: Uploaded image file in any PIL-supported format

    Returns:
        StreamingResponse: PNG image with colorization applied

    Raises:
        HTTPException: 415 if image file is invalid
        HTTPException: 501 if selected model is not available
    """
    if model == ModelType.VIT:
        raise HTTPException(
            status_code=501, detail="ViT model not trained yet – check back later!"
        )

    if model == ModelType.QUANT:
        raise HTTPException(
            status_code=501, detail="QUANT model not trained yet – check back later!"
        )

    try:
        image.file.seek(0)
        pil_img = Image.open(image.file)
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
    except PIL.UnidentifiedImageError:
        raise HTTPException(
            status_code=415,
            detail="Invalid image file. Please upload a valid image in JPEG, PNG, TIFF, BMP, WebP, or other PIL-supported format.",
        )

    out_img = MODEL_HUB.colorize_with_resnet(pil_img)

    buffer = io.BytesIO()
    out_img.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
