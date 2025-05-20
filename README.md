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
│
├── app/                    # Streamlit web application
│   ├── __init__.py         # App initialization
│   ├── app.py              # Main Streamlit UI
│   ├── model_loader.py     # Model loading logic
│   └── utils.py            # UI utility functions
│
├── data/                   # Image datasets
│   ├── train2017/          # Training images
│   ├── val2017/            # Validation images
│   └── test2017/           # Test/demo images
│
├── src/                    # Core source code
│   ├── configs/            # YAML configuration files
│   ├── dataloaders/        # Dataset preparation code
│   ├── models/             # Model architecture
│   │   ├── base_model.py   # Abstract model class
│   │   └── resnet.py       # ResNet colorization model
│   ├── pipelines/          # Training and evaluation pipelines
│   └── utils/              # Generic utilities
│
├── .python-version         # Python version file
├── pyproject.toml          # Project and dependency manager config
└── README.md               # 📄 You're here!
```


## 🚀 Usage
### 🖼️ Running the Web Application
To start the Streamlit web interface:
```bash
cd app
streamlit run app.py
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
   
