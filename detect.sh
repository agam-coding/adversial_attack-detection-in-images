#!/bin/bash

# Simple wrapper script for the adversarial attack detector

# Set PYTHONPATH to include the current directory
export PYTHONPATH=.

# Check if an image path is provided
if [ "$1" == "--setup" ]; then
    echo "Setting up sample images..."
    python3 src/download_samples.py
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Adversarial Attack Detector"
    echo ""
    echo "Usage:"
    echo "  ./detect.sh                       - Run in interactive mode"
    echo "  ./detect.sh --image <image_path>  - Analyze a specific image"
    echo "  ./detect.sh --evaluate            - Run evaluation on test data"
    echo "  ./detect.sh --setup               - Setup sample test images"
    echo "  ./detect.sh --upload <image_path> - Upload and analyze an image"
    echo "  ./detect.sh --dataset             - Browse and analyze Fashion MNIST dataset"
    echo "  ./detect.sh --help                - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./detect.sh --image test_images/adversarial_tshirt.png"
    echo "  ./detect.sh --upload /path/to/your/image.jpg --copy"
    echo "  ./detect.sh --dataset --list 20"
elif [ "$1" == "--upload" ]; then
    shift  # Remove the --upload argument
    if [ -z "$1" ]; then
        echo "Error: Please provide a path to an image."
        echo "Usage: ./detect.sh --upload <image_path> [--copy]"
        exit 1
    fi
    
    # Collect all arguments to pass to the upload script
    args="$@"
    
    echo "Uploading and analyzing image..."
    python3 src/upload_image.py $args
elif [ "$1" == "--dataset" ]; then
    shift  # Remove the --dataset argument
    
    echo "Browsing Fashion MNIST dataset..."
    python3 src/detect_from_dataset.py "$@"
else
    # Pass all arguments to the Python script
    python3 src/detect_adversarial.py "$@"
fi 