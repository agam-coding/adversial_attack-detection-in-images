#!/bin/bash

# Script to analyze a directory of images with custom visualization

# Ensure PYTHONPATH is set correctly
export PYTHONPATH=.

echo "================================================================"
echo "  Adversarial Attack Detector - Custom Visualization"
echo "================================================================"
echo ""

# Help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage:"
    echo "  ./analyze_directory.sh /path/to/image/dir    - Analyze images in directory"
    echo "  ./analyze_directory.sh path/to/single/image.jpg - Analyze a single image"
    echo ""
    echo "This tool will analyze images for adversarial attacks and create"
    echo "customized visualizations showing:"
    echo "  1. Original image"
    echo "  2. Image with perturbations highlighted"
    echo "  3. Detection scores: noise reduction, perturbation size, confidence score"
    echo ""
    echo "Results will be saved to the custom_results/ directory."
    exit 0
fi

# Check if a path was provided
if [ -z "$1" ]; then
    echo "Please provide a path to an image or directory."
    echo "Run with --help for usage information."
    exit 1
fi

# Create custom_results directory if it doesn't exist
mkdir -p custom_results

# Check if the path is a directory or a single image
if [ -d "$1" ]; then
    echo "Processing directory: $1"
    # Process the entire directory
    PYTHONPATH=. python src/custom_visualization.py --dir "$1"
else
    echo "Processing single image: $1"
    # Process a single image
    PYTHONPATH=. python src/custom_visualization.py --image "$1"
fi

echo ""
echo "Analysis complete. Results saved to custom_results/ directory."
echo "================================================================" 