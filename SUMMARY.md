# Adversarial Attack Detector - Feature Summary

We've created a comprehensive tool for detecting adversarial attacks on images. This summary outlines all available features.

## Available Features

### 1. Adversarial Attack Detection

The core functionality analyzes images to determine if they have been manipulated to fool neural networks:

- Uses 5 different detection methods
- Provides a risk score (0-1 scale)
- Gives a clear assessment: ✅ NO ADVERSARIAL ATTACK DETECTED or ❌ ADVERSARIAL ATTACK DETECTED
- Visualizes the detection results with detailed scores

### 2. Multiple Input Methods

Users can analyze images in several ways:

- **Interactive Mode**: Simple drag-and-drop interface for any image file
- **Upload Utility**: Copies images to the project and analyzes them
- **Command Line**: Direct analysis of specified images
- **Dataset Browser**: Browse and select images from the Fashion MNIST dataset or analyze your own images

### 3. Dataset Interaction

The dataset browser provides:

- Ability to view sample images from the Fashion MNIST dataset
- Analysis of specific dataset samples
- Generation of adversarial versions for comparison
- Side-by-side comparison of original and adversarial images
- **Analysis of your own images from any directory** on your computer

### 4. Evaluation Tools

For more technical users:

- Performance evaluation on multiple samples
- ROC and precision-recall curve generation
- Comparison of detection performance on clean vs. adversarial images

## Usage Options

| Feature | Command | Description |
|---------|---------|-------------|
| Interactive Mode | `./detect.sh` | Prompts for image path with drag & drop support |
| Analyze Image | `./detect.sh --image path/to/image.jpg` | Analyze specific image file |
| Upload & Analyze | `./detect.sh --upload path/to/image.jpg --copy` | Copy to project and analyze |
| Dataset Browser | `./detect.sh --dataset` | Interactive dataset browsing and analysis |
| List Dataset Samples | `./detect.sh --dataset --list 20` | Generate 20 sample images from dataset |
| Analyze Dataset Sample | `./detect.sh --dataset --sample 3` | Analyze sample #3 from dataset |
| Generate Adversarial | `./detect.sh --dataset --sample 3 --adversarial` | Create & analyze adversarial version |
| Evaluation | `./detect.sh --evaluate --samples 10` | Test on 10 clean & adversarial samples |

### Multiple Ways to Analyze Your Own Images

The tool provides several options to analyze your own images:

1. **Interactive Mode**: `./detect.sh`
2. **Command Line**: `./detect.sh --image /path/to/your/image.jpg`
3. **Upload Utility**: `./detect.sh --upload /path/to/your/image.jpg --copy`
4. **Dataset Browser**: `./detect.sh --dataset` → Option 5: "Analyze your own image from any directory"

Each method provides the same comprehensive analysis with clear adversarial attack detection results.

## Documentation

The project includes several guides:

- `README.md` - Main documentation
- `UPLOAD_GUIDE.md` - Quick guide for uploading and analyzing images
- `USER_GUIDE.md` - Detailed instructions with troubleshooting tips
- `dataset_samples/README.md` - Guide for the dataset browser
- `my_images/README.md` - Instructions for using your own images
- `test_images/README.md` - Information about test images 