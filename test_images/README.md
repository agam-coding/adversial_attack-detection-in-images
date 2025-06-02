# Test Images for Adversarial Attack Detection

This directory contains sample images for testing the adversarial attack detector:

1. `clean_tshirt.png` - A synthetic clean Fashion MNIST-like image
2. `adversarial_tshirt.png` - A version with subtle adversarial modifications
3. `comparison.png` - Side-by-side comparison of clean and adversarial versions
4. Other sample images downloaded from the internet

## Testing Instructions

Run the detector on these images:

```bash
# Interactive mode
PYTHONPATH=. python3 src/detect_adversarial.py

# Or directly specify an image
PYTHONPATH=. python3 src/detect_adversarial.py --image test_images/adversarial_tshirt.png
```

The detector should identify the adversarial image as suspicious.
