#!/bin/bash

# Simple helper script for analyzing your own images

echo "==============================================="
echo "  Adversarial Attack Detector - Image Analyzer"
echo "==============================================="
echo ""
echo "This script will help you analyze your own image for adversarial attacks."
echo ""

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage:"
    echo "  ./analyze_my_image.sh                    - Interactive prompt to enter image path"
    echo "  ./analyze_my_image.sh path/to/image.jpg  - Directly analyze specific image"
    echo ""
    echo "TIP: You can drag and drop an image file from your file explorer into the terminal"
    echo "     instead of typing the full path."
    exit 0
fi

if [ -n "$1" ]; then
    # If an argument is provided, use it as the image path
    IMAGE_PATH="$1"
    
    # Check if the file exists
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "Error: Image file not found at: $IMAGE_PATH"
        echo "Please check the path and try again."
        exit 1
    fi
    
    echo "Analyzing image: $IMAGE_PATH"
    ./detect.sh --image "$IMAGE_PATH"
else
    # No argument, prompt the user
    echo "Please enter the path to your image file."
    echo "TIP: You can drag and drop the image file into this terminal window."
    echo ""
    read -p "Image path: " IMAGE_PATH
    
    if [ -z "$IMAGE_PATH" ]; then
        echo "No path entered. Exiting."
        exit 1
    fi
    
    # Check if the file exists
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "Error: Image file not found at: $IMAGE_PATH"
        echo "Please check the path and try again."
        exit 1
    fi
    
    echo "Analyzing image: $IMAGE_PATH"
    ./detect.sh --image "$IMAGE_PATH"
fi 