#!/bin/bash

# Change to the project root directory
cd "$(dirname "$0")/.."

# Ensure required directories exist
mkdir -p app/uploaded_images

# Run the Streamlit app
streamlit run app/app.py 