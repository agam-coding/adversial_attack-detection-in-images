#!/bin/bash

# Interactive script for custom adversarial analysis visualization

# Ensure PYTHONPATH is set correctly
export PYTHONPATH=.

echo "================================================================"
echo "  Adversarial Attack Detector - Interactive Custom Analysis"
echo "================================================================"
echo ""
echo "This tool helps you analyze images for adversarial attacks with a"
echo "customized visualization showing:"
echo "  1. Original image"
echo "  2. Image with perturbations highlighted"
echo "  3. Detection scores: noise reduction, perturbation size, confidence"
echo ""

# Create custom_results directory
mkdir -p custom_results

# Interactive mode
echo "Please enter the path to your image or directory."
echo "TIP: You can drag and drop the file/folder from your file explorer."
echo ""
read -p "Path: " USER_PATH

if [ -z "$USER_PATH" ]; then
    echo "No path provided. Exiting."
    exit 1
fi

# Check if path exists
if [ ! -e "$USER_PATH" ]; then
    echo "Error: Path does not exist: $USER_PATH"
    exit 1
fi

echo ""
echo "Processing: $USER_PATH"
echo ""

# Process based on whether it's a directory or file
if [ -d "$USER_PATH" ]; then
    echo "Analyzing all images in directory: $USER_PATH"
    PYTHONPATH=. python src/custom_visualization.py --dir "$USER_PATH"
else
    echo "Analyzing image: $USER_PATH"
    PYTHONPATH=. python src/custom_visualization.py --image "$USER_PATH"
fi

echo ""
echo "Analysis complete! Results saved to custom_results/ directory."
echo "================================================================" 