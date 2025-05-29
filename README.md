# 🎨 Automatic Image Colorization

A deep learning project that brings black-and-white photos to life by colorizing them using a ResNet-based neural network. Featuring an interactive web interface, this tool allows you to explore automatic image colorization from your browser.

---

## 🧠 Overview

This project uses a convolutional neural network based on **ResNet** to colorize grayscale images. The model predicts the `a` and `b` color channels in the **Lab color space**, given only the lightness (`L`) channel.

---

## ✨ Features

- 🖥️ **Interactive Web UI** – Powered by Streamlit for real-time image colorization  
- 🪞 **Side-by-Side Comparison** – Interactive slider to compare original and colorized images  
- 🧪 **Multiple Models** – Plug-and-play support for different colorization models (e.g., dummy model, ResNet)  
- 💾 **Download Results** – Save your colorized images for later use or sharing  

---

## ⚙️ Setup & Installation

### 📦 Step 1: Install `uv` (if not installed)
```bash
pip install uv
```

### 📥 Step 2: Clone the repository:
```bash
git clone https://github.com/atodorov284/image_colorizer.git
cd image_colorizer
```

### 🔧 Step 3: Install Dependencies
```bash
uv sync
```

## 📁 Project Structure
```
image_colorizer/
├── README.md                           # Project documentation
├── app/                               # Streamlit web application
│   ├── __init__.py
│   ├── app.py                         # Main Streamlit UI interface
│   ├── model_loader.py                # Handles loading trained models
│   └── utils.py                       # UI utility functions and helpers
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── data_analysis.ipynb            # Dataset exploration and statistics
│   └── filtering.ipynb                # Data preprocessing experiments
├── pyproject.toml                     # Project dependencies and metadata
├── src/                              # Core source code
│   ├── __init__.py
│   ├── api/                          # API-related modules
│   │   ├── __init__.py
│   │   ├── front_end.py              
│   │   ├── main.py                   # Main API application
│   │   └── model_hub.py              # Model management and registry
│   ├── configs/
│   │   └── resnet_config.yaml        # ResNet model configuration parameters
│   ├── dataloaders/
│   │   ├── __init__.py
│   │   └── colorization_dataset.py   # Dataset loading and preprocessing
│   ├── models/                       # Neural network architectures
│   │   ├── __init__.py
│   │   ├── base_model.py             # Abstract base class for all models
│   │   ├── resnet.py                 # ResNet-based colorization model
│   │   └── vit.py              
│   ├── pipelines/                    # Training and inference workflows
│   │   ├── __init__.py
│   │   ├── base_pipeline.py          # Abstract pipeline base class
│   │   └── colorization_pipeline.py  # Complete colorization workflow
│   ├── predict.py                    # Standalone prediction script
│   ├── train.py                      # Model training script
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── colorization_utils.py     # Color space conversion utilities
│       ├── early_stopping.py        # Training early stopping logic
│       ├── filtering_utils.py        # Image filtering and preprocessing
│       └── predicting_utils.py       # Prediction helper functions
└── uv.lock                           # Locked dependency versions
```


## 🚀 Usage
### 🖼️ Running the Web Application

To run the API:
```bash
cd src
uvicorn api.main:app --reload
```
- Upload an image to predict/resnet
- Or upload an image to predict/vit (development still in progress)

To start the Streamlit web interface:
```bash
streamlit run app/app.py
```
This will launch the application in your default browser. You can:

- Upload an image to colorize, or use one of the provided example images
- Select a colorization model from the sidebar
- View the results with an interactive before/after slider
- Download your colorized image

## 🏋️‍♂️ Training a New Model

To train a custom **ResNet-based** colorization model:

1. 📁 **Prepare your dataset** and place it under the `data/` directory.

2. ⚙️ **Update the configuration** file:
   ```bash
   src/configs/resnet_config.yaml
   ```
3. 🚀 **Run the training script**:
   ```bash
   python src/train.py
   ```
   
