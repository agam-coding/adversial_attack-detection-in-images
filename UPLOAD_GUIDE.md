# Input & Check: Analyze Your Own Images for Adversarial Attacks

This guide shows you how to input your own images and have our AI model check them for signs of adversarial attacks.

## How It Works: Simple Input → Check → Result

1. **INPUT**: You provide any image from your computer
2. **CHECK**: Our AI model analyzes it using multiple detection methods
3. **RESULT**: You get a clear verdict: ✅ CLEAN or ❌ ADVERSARIAL

## Quickest Method: Direct Analysis Script

We've created a dedicated script just for analyzing your images:

```bash
./analyze_my_image.sh
```

When prompted, simply drag and drop your image file from your file explorer into the terminal, and press Enter.

Or directly specify your image path:

```bash
./analyze_my_image.sh /path/to/your/image.jpg
```

## Other Ways to Input & Check Your Images

### Method 1: Drag & Drop (Interactive Mode)

The classic approach:

1. Run:
   ```bash
   ./detect.sh
   ```
2. When prompted, **drag and drop** your image file from your file explorer into the terminal
3. Press Enter
4. View the AI's analysis results!

### Method 2: Upload & Check 

To save a copy of your image while analyzing it:

```bash
./detect.sh --upload /path/to/your/image.jpg --copy
```

This will:
1. Copy your image to the `my_images` folder for safekeeping
2. Run the AI detector on your image
3. Tell you if an adversarial attack was detected

### Method 3: Dataset Browser with Your Images

To analyze your images alongside dataset samples:

1. Run:
   ```bash
   ./detect.sh --dataset
   ```
2. Select option 5: "Analyze your own image from any directory"
3. Enter your image path or drag & drop the file
4. View the analysis results!

## Understanding Your Results

After the AI checks your image, you'll see results like this:

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

### How to Interpret the Results:

- **Risk Score**: This is the key metric (scale 0-1)
  - Score < 0.5: ✅ NO ADVERSARIAL ATTACK DETECTED (clean image)
  - Score ≥ 0.5: ❌ ADVERSARIAL ATTACK DETECTED (manipulated image)

- **Visualization**: The tool creates a detailed visualization showing:
  - Your original image
  - The scores for each detection method
  - An overall assessment

## Tips for Successful Input

- **Drag & Drop**: The easiest way to avoid path errors
- **Any Format**: The tool handles JPG, PNG, BMP, and other image formats
- **Any Size**: Images are automatically resized to match the model's requirements
- **Path with Spaces**: Use quotes if your path contains spaces:
  ```bash
  ./analyze_my_image.sh "/Users/name/Photos/my vacation.jpg"
  ```

## Where to Find Your Results

All analysis results are saved in the `docs` directory with clear filenames that match your input image.

For example, if you analyzed "vacation.jpg", look for "detection_vacation_*.png" in the docs folder.

These visualizations are perfect for sharing or further analysis! 