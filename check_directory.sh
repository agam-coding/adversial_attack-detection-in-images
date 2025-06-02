#!/bin/bash

# Script to check all images in a specified directory for adversarial attacks

# Ensure PYTHONPATH is set correctly
export PYTHONPATH=.

echo "================================================================"
echo "  Adversarial Attack Detector - Directory Scanner"
echo "================================================================"
echo ""

# Help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage:"
    echo "  ./check_directory.sh                       - Interactive prompt for directory"
    echo "  ./check_directory.sh /path/to/image/dir    - Check all images in specified directory"
    echo ""
    echo "This tool will scan all image files in the specified directory and"
    echo "check them for signs of adversarial attacks."
    echo ""
    echo "Each image will be analyzed with advanced visualization showing:"
    echo "  - Original and processed image"
    echo "  - Detection scores from multiple methods"
    echo "  - Overall risk assessment"
    exit 0
fi

# Function to check if a file is an image
is_image_file() {
    local file="$1"
    local ext="${file##*.}"
    ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
    
    # Check for common image extensions
    if [[ "$ext" == "jpg" || "$ext" == "jpeg" || "$ext" == "png" || 
          "$ext" == "bmp" || "$ext" == "tiff" || "$ext" == "gif" ]]; then
        return 0  # True - is an image
    else
        return 1  # False - not an image
    fi
}

# Function to analyze a single image using the new visualization
analyze_image() {
    local image_path="$1"
    local output_dir="$2"
    local base_name=$(basename "$image_path")
    local result_file="${output_dir}/result_${base_name%.*}.txt"
    local visual_file="${output_dir}/visual_${base_name%.*}.png"
    
    echo "Analyzing: $image_path"
    
    # Run the detector with our new visualization script
    # Ensure PYTHONPATH is set for each command
    PYTHONPATH=. python src/visualize_detection.py --image "$image_path" --save "$visual_file" > "$result_file"
    
    # Extract the risk score and assessment from the output
    local risk_score=$(grep "Risk Score:" "$result_file" | awk '{print $3}')
    local assessment=$(grep "Assessment:" "$result_file" | awk '{print $2}')
    
    if [[ "$assessment" == "LIKELY" ]]; then
        echo "  ❌ LIKELY ADVERSARIAL (Risk score: ${risk_score})"
        echo "$image_path" >> "${output_dir}/adversarial_images.txt"
    else
        echo "  ✅ LIKELY CLEAN (Risk score: ${risk_score})"
        echo "$image_path" >> "${output_dir}/clean_images.txt"
    fi
}

# Directory to scan
DIR_PATH=""

# Get directory path - either from command line or prompt
if [ -n "$1" ]; then
    DIR_PATH="$1"
else
    echo "Please enter the path to your image directory."
    echo "TIP: You can drag and drop the directory from your file explorer."
    echo ""
    read -p "Directory path: " DIR_PATH
fi

# Remove trailing slash if present
DIR_PATH=${DIR_PATH%/}

# Check if directory exists
if [ ! -d "$DIR_PATH" ]; then
    echo "Error: Directory not found at: $DIR_PATH"
    echo "Please check the path and try again."
    exit 1
fi

echo ""
echo "Scanning directory: $DIR_PATH"

# Create results directory
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RESULTS_DIR="scan_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR/"
echo ""

# Initialize counters and result files
TOTAL_FILES=0
IMAGE_FILES=0
ADVERSARIAL_COUNT=0
CLEAN_COUNT=0

echo "" > "${RESULTS_DIR}/adversarial_images.txt"
echo "" > "${RESULTS_DIR}/clean_images.txt"

# List all files in the directory
echo "Identifying image files..."
for file in "$DIR_PATH"/*; do
    if [ -f "$file" ]; then
        TOTAL_FILES=$((TOTAL_FILES + 1))
        
        if is_image_file "$file"; then
            IMAGE_FILES=$((IMAGE_FILES + 1))
            echo "$file" >> "${RESULTS_DIR}/image_files.txt"
        fi
    fi
done

echo "Found $IMAGE_FILES image files out of $TOTAL_FILES total files."
echo ""

# Check if any image files were found
if [ "$IMAGE_FILES" -eq 0 ]; then
    echo "No image files found in the directory. Exiting."
    exit 0
fi

echo "====== Starting Analysis ======"
echo ""

# Process each image file
while read -r image_file; do
    analyze_image "$image_file" "$RESULTS_DIR"
done < "${RESULTS_DIR}/image_files.txt"

# Count results
ADVERSARIAL_COUNT=$(grep -c . "${RESULTS_DIR}/adversarial_images.txt")
CLEAN_COUNT=$(grep -c . "${RESULTS_DIR}/clean_images.txt")

echo ""
echo "====== Analysis Complete ======"
echo ""
echo "Summary:"
echo "  Total images analyzed: $IMAGE_FILES"
echo "  ✅ Clean images: $CLEAN_COUNT"
echo "  ❌ Adversarial images: $ADVERSARIAL_COUNT"
echo ""
echo "Detailed results saved to: $RESULTS_DIR/"
echo "  - Advanced visual reports for each image"
echo "  - List of clean and adversarial images"
echo ""
echo "================================================================" 