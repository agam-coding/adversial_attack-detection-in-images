#!/bin/bash

# Script to run the Streamlit app for Adversarial Attack Detection

# Ensure required packages are installed
echo "Checking for required packages..."
pip install -q streamlit matplotlib tensorflow pillow

# Set PYTHONPATH environment variable
export PYTHONPATH=.

echo "Starting Adversarial Attack Detector app..."
echo "View the app in your browser when it launches."
echo ""

# Run the Streamlit app
streamlit run src/streamlit_app.py 