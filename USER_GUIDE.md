# Comprehensive Guide: Input Images & Check for Adversarial Attacks

This comprehensive guide will walk you through all the ways to input your own images and have our AI model check them for adversarial attacks.

## Quick Start: Input → Check → Result

The simplest workflow:

1. Run our dedicated analysis script:
   ```bash
   ./analyze_my_image.sh
   ```

2. Drag and drop your image into the terminal when prompted

3. Get immediate results: ✅ CLEAN or ❌ ADVERSARIAL

## Input Methods: Choose What Works for You

We've created multiple ways for you to input your images:

### 1. Prepare Your Image

- You can use **any image** from your computer
- All formats are supported (JPG, JPEG, PNG, BMP, etc.)
- Any size works - our tool automatically handles resizing

### 2. Choose Your Input Method

#### Option A: One-Step Analysis Script (Recommended)

1. Run:
   ```bash
   ./analyze_my_image.sh
   ```
   
2. When prompted, drag and drop your image file from your file explorer directly into the terminal

#### Option B: Interactive Mode

1. Run:
   ```bash
   ./detect.sh
   ```
   
2. At the prompt, drag and drop your image or type its full path

#### Option C: Command Line (Direct)

For a one-line approach:
```bash
./detect.sh --image /path/to/your/image.jpg
```

#### Option D: Upload & Copy

To save a copy of your image while analyzing it:
```bash
./detect.sh --upload /path/to/your/image.jpg --copy
```

#### Option E: Dataset Browser with Custom Images

1. Run:
   ```bash
   ./detect.sh --dataset
   ```
   
2. Select option 5: "Analyze your own image from any directory"

3. Enter your image path or drag & drop the file

### 3. Tips for Successful Image Input

- **Drag & Drop Method**: The most reliable way to input images without path errors
- **Full Paths**: Always use the complete path to your image if typing manually
- **Paths with Spaces**: If your path contains spaces, enclose it in quotes:
  ```
  "/Users/name/Desktop/my vacation.jpg"
  ```
- **Windows Paths**: Use forward slashes even on Windows: `C:/Users/name/Pictures/image.jpg`
- **Relative Paths**: You can use relative paths if you're in the project directory:
  ```
  my_images/sample.jpg
  ```

## Understanding the AI's Check Results

After checking your image, you'll receive comprehensive results:

### Visual Results

The AI generates a visual report saved in the `docs` directory showing:
- Your original image
- Detection scores for each method
- Overall risk assessment

### Text Results

You'll see a summary like this:
```
==================================================
ADVERSARIAL ATTACK DETECTION RESULTS
==================================================
Image analyzed: your_image.jpg
Predicted class: Sneaker (Class 7)
Confidence: 0.1153
Risk score: 0.5155 (Threshold: 0.5)

❌ ALERT: ADVERSARIAL ATTACK DETECTED
This image shows signs of adversarial manipulation intended to fool neural networks.

Detailed visualization saved to: docs/detection_result.png
==================================================
```

### Interpreting the Results

The key metric is the **Risk Score**:
- Score < 0.5: ✅ NO ADVERSARIAL ATTACK DETECTED (clean image)
- Score ≥ 0.5: ❌ ADVERSARIAL ATTACK DETECTED (manipulated image)

Each detection method contributes to this score:
- **Confidence Score**: Lower confidence often indicates adversarial manipulation
- **Prediction Consistency**: Changes with small noise suggest adversarial tampering
- **Feature Squeezing**: Changes after color depth reduction reveal attacks
- **Perturbation Size**: Measures possible adversarial noise
- **Noise Reduction**: Tests if denoising changes predictions

## Common Issues and Solutions

### "File not found" Error

If you see this error:
- Try the drag & drop method instead of typing the path
- Make sure you're using the full path to the file
- Check for typos in the filename or directory

### "Error processing image" 

If you get this error:
- Try a different image format (convert PNG to JPG, etc.)
- Make sure the file isn't corrupted
- Check if the image is readable by other programs

## Sample Images for Testing

We've included test images you can use to try the detector:

- `test_images/clean_tshirt.png`: A clean image
- `test_images/adversarial_tshirt.png`: An image with adversarial modifications
- `test_images/comparison.png`: Side-by-side comparison

Try analyzing both to see how the detector differentiates between them! 