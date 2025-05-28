# Image Colorization

## Project Structure

```
├── README.md
├── app
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   ├── app.cpython-39.pyc
│   │   ├── model_loader.cpython-39.pyc
│   │   └── utils.cpython-39.pyc
│   ├── app.py
│   ├── model_loader.py
│   └── utils.py
├── notebooks
│   ├── data_analysis.ipynb
│   └── filtering.ipynb
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-39.pyc
│   ├── api
│   │   ├── __init__.py
│   │   ├── front_end.py
│   │   └── main.py
│   ├── configs
│   │   └── resnet_config.yaml
│   ├── dataloaders
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-39.pyc
│   │   │   └── colorization_dataset.cpython-39.pyc
│   │   └── colorization_dataset.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-39.pyc
│   │   │   ├── base_model.cpython-39.pyc
│   │   │   └── resnet.cpython-39.pyc
│   │   ├── base_model.py
│   │   ├── resnet.py
│   │   └── vit.py
│   ├── pipelines
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-39.pyc
│   │   │   ├── base_pipeline.cpython-39.pyc
│   │   │   └── colorization_pipeline.cpython-39.pyc
│   │   ├── base_pipeline.py
│   │   └── colorization_pipeline.py
│   ├── predict.py
│   ├── train.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-39.pyc
│       │   ├── colorization_utils.cpython-39.pyc
│       │   ├── early_stopping.cpython-39.pyc
│       │   └── filtering_utils.cpython-39.pyc
│       ├── colorization_utils.py
│       ├── early_stopping.py
│       ├── filtering_utils.py
│       └── predicting_utils.py
└── uv.lock
```

## Description



## Installation

Add installation instructions here...

## Usage

Add usage instructions here...
