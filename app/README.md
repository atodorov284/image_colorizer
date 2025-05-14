# Image Colorization App

This Streamlit app demonstrates the image colorization functionality of the project. It provides a user-friendly interface to upload grayscale images or use example images and see them colorized in real-time.

## Features

- Upload your own images or use example images from the test dataset
- Adjust image size for processing
- View side-by-side comparison of original, grayscale, and colorized images
- Download colorized results
- Real-time colorization using a trained deep learning model
- Fallback to demo mode if no model is available

## How to Run the App

### Option 1: Using the run script (recommended)

The easiest way to run the app is to use the provided shell script:

```bash
./app/run_app.sh
```

Or use the Python entry point:

```bash
python app_entry.py
```

### Option 2: Using Streamlit directly

1. Ensure you have all dependencies installed:
```bash
pip install -r app/requirements.txt
```

2. Navigate to the project root directory:
```bash
cd /path/to/project
```

3. Run the Streamlit app:
```bash
streamlit run app/app.py
```

4. The app will open in your default web browser at http://localhost:8501

## Interface Guide

- **Upload Image**: Click "Browse files" to upload your own image
- **Use Example Image**: Click to use a sample image from the test dataset
- **Image Size**: Adjust the slider to change the processing size (larger sizes may take more time)
- **Model Selection**: Choose between different colorization models (when available)

## How It Works

1. The image is converted to grayscale (L channel in the LAB color space)
2. The grayscale image is fed into a trained neural network (ResNet)
3. The network predicts the color channels (A and B in the LAB color space)
4. The predicted color channels are combined with the original L channel
5. The resulting LAB image is converted back to RGB for display

## Demo Mode

If no trained model is available, the app will run in demo mode, which:
- Uses a dummy colorizer that generates random colorization
- Still provides the full interface for testing and interaction
- Allows you to experience the app workflow while you work on training models

## Troubleshooting

- If the app fails to start, ensure you're running from the project root directory
- If you see import errors, ensure all required packages are installed:
  ```bash
  pip install -r app/requirements.txt
  ```
- The app includes fallbacks for most errors, so it should run in demo mode even with missing dependencies
- For best results, use well-lit images with clear subjects 