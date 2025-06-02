# Simple Guide: Input & Check Your Images

This is the simplest way to check your images for adversarial attacks.

## 1-2-3 Process

1. **RUN** this script:
   ```
   ./analyze_my_image.sh
   ```

2. **INPUT** your image:
   - Simply drag & drop your image file into the terminal
   - Or type the path to your image

3. **CHECK** the results:
   - ✅ **CLEAN**: No adversarial attack detected
   - ❌ **ADVERSARIAL**: Adversarial attack detected

That's it!

## Example

```
$ ./analyze_my_image.sh

===============================================
  Adversarial Attack Detector - Image Analyzer
===============================================

This script will help you analyze your own image for adversarial attacks.

Please enter the path to your image file.
TIP: You can drag and drop the image file into this terminal window.

Image path: [drag & drop your image here]

Analyzing image: /Users/name/Pictures/my_image.jpg
...

==================================================
ADVERSARIAL ATTACK DETECTION RESULTS
==================================================

Risk score: 0.4883 (Threshold: 0.5)

✅ NO ADVERSARIAL ATTACK DETECTED
This image appears to be clean with no signs of adversarial manipulation.

Detailed visualization saved to: docs/detection_result.png
==================================================
```

## More Options

To analyze a specific image directly:

```
./analyze_my_image.sh /path/to/your/image.jpg
```

See the full guides for more details:
- `UPLOAD_GUIDE.md` - More ways to input & check images
- `USER_GUIDE.md` - Comprehensive instructions 