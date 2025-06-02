#!/bin/bash

# Simple wrapper script for the upload_image.py script

# Ensure PYTHONPATH is set correctly for importing modules
export PYTHONPATH=.

# Ensure docs directory exists for README previews
mkdir -p docs

echo "====================================================="
echo "   Adversarial Attack Detector - Upload & Check"
echo "====================================================="
echo ""
echo "This tool helps you analyze your own images for adversarial attacks."
echo "Results will be displayed in an advanced visualization format."
echo ""

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage:"
    echo "  ./upload_and_check.sh                     - Interactive mode"
    echo "  ./upload_and_check.sh path/to/image.jpg   - Analyze specific image"
    echo "  ./upload_and_check.sh --copy path/to/image.jpg - Copy and analyze image"
    echo ""
    echo "TIP: You can drag and drop an image from your file explorer into the terminal"
    echo "     instead of typing the full path."
    echo ""
    echo "The results will show:"
    echo "  - Original image and classified result"
    echo "  - Detection scores for various adversarial detection methods"
    echo "  - Overall risk assessment (LIKELY ADVERSARIAL or LIKELY CLEAN)"
    echo ""
    echo "Visualization will be saved to the docs/ directory."
    exit 0
fi

if [ "$1" == "--copy" ]; then
    shift
    # Explicitly set PYTHONPATH for each command
    PYTHONPATH=. python src/upload_image.py --copy "$@"
else
    # Explicitly set PYTHONPATH for each command
    PYTHONPATH=. python src/upload_image.py "$@"
fi 