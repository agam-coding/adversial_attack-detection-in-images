# Directory Scanner: Check Multiple Images At Once

This guide explains how to scan an entire directory of images for adversarial attacks in one go.

## Quick Start

1. Run the directory scanner:
   ```bash
   ./check_directory.sh
   ```

2. When prompted, enter the path to your image directory:
   ```
   Directory path: /path/to/your/images
   ```
   
   TIP: You can drag & drop a folder from your file explorer directly into the terminal.

3. Wait for the analysis to complete and view the results.

## Direct Method

You can also specify the directory directly:

```bash
./check_directory.sh /path/to/your/images
```

## What Happens

1. The scanner identifies all image files in the directory
2. Each image is analyzed for signs of adversarial attacks
3. A summary of results is displayed:
   - Total number of images analyzed
   - Number of clean images (no attack detected)
   - Number of adversarial images (attack detected)

4. Detailed results are saved to a timestamped directory (`scan_results_YYYYMMDD_HHMMSS`)

## Example Output

```
================================================================
  Adversarial Attack Detector - Directory Scanner
================================================================

Scanning directory: /Users/name/Pictures/test_images

Results will be saved to: scan_results_20230615_143022/

Identifying image files...
Found 12 image files out of 15 total files.

====== Starting Analysis ======

Analyzing: /Users/name/Pictures/test_images/image1.jpg
  ✅ NO ADVERSARIAL ATTACK DETECTED (Risk score: 0.4215)
Analyzing: /Users/name/Pictures/test_images/image2.png
  ❌ ADVERSARIAL ATTACK DETECTED (Risk score: 0.6732)
...

====== Analysis Complete ======

Summary:
  Total images analyzed: 12
  ✅ Clean images: 10
  ❌ Adversarial images: 2

Detailed results saved to: scan_results_20230615_143022/
  - Visual reports for each image
  - List of clean and adversarial images

================================================================
```

## Result Files

The results directory contains:
- `image_files.txt` - List of all image files analyzed
- `clean_images.txt` - List of images with no adversarial attack detected
- `adversarial_images.txt` - List of images with adversarial attack detected
- `visual_*.png` - Visual report for each image
- `result_*.txt` - Detailed analysis for each image

## Notes

- Supported image formats: JPG, JPEG, PNG, BMP, TIFF, GIF
- The scanner will ignore non-image files
- The analysis uses the same detection methods as the single-image analyzer 