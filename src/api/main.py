from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse
import io
import PIL
from PIL import Image, ImageOps

from .model_hub import ModelHub

MODEL_HUB = ModelHub()

app = FastAPI(
    title="Grayscale → Colour API",
    summary="Upload a grayscale image, receive a colourised PNG.",
    description="""
This endpoint wraps any **PyTorch** colourisation network that implements a
`.predict(tensor)` or `.forward(tensor)` and returns an RGB NumPy array.

### Request
* **POST /colourise**  
  Multipart-form upload with key **`image`**.

### Response
* **200** ⇒ `image/png` stream.

### Caveats
* The model has been trained on 256 × 256 crops; other sizes are resized.
""",
    version="0.1.0",
)

@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')

@app.post(
    "/predict/resnet",
    description="Colourise an image using the pretrained ResNet‑18 model.",
    responses={
        200: {"content": {"image/png": {}}},
        415: {"description": "Invalid image supplied."},
    },
)
async def predict_resnet(image: UploadFile):
    """Return a PNG image produced by the ResNet colourisation model."""
    try:
        image.file.seek(0)                     # safety: start of stream
        pil_img = Image.open(image.file)
        pil_img = ImageOps.exif_transpose(pil_img)  # ← key line
        pil_img = pil_img.convert("RGB")
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image file.")

    out_img = MODEL_HUB.colorize_with_resnet(pil_img)

    buffer = io.BytesIO()
    out_img.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post(
    "/predict/vit",
    description="Placeholder – ViT model is not yet trained.",
    responses={501: {"description": "Model not available."}},
    )
async def predict_vit(image: UploadFile):
    """ViT endpoint isn’t ready yet."""
    raise HTTPException(
        status_code=501,
        detail="ViT model not trained yet – check back later!"
    )