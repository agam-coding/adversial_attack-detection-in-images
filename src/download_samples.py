import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Create directories
def create_directories():
    os.makedirs('test_images', exist_ok=True)
    print("Created test_images directory")

# Download a sample image from a URL
def download_image(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(f"test_images/{filename}", 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

# Generate fashion MNIST like images for testing
def generate_sample_images():
    # Create synthetic FMNIST-like images
    print("Generating sample Fashion MNIST-like images...")
    
    # Generate a clean t-shirt image
    clean = np.zeros((28, 28), dtype=np.float32)
    # Draw a simple t-shirt shape
    clean[5:7, 9:19] = 0.8  # collar
    clean[7:20, 7:21] = 0.5  # body
    clean[7:15, 5:7] = 0.6  # left sleeve
    clean[7:15, 21:23] = 0.6  # right sleeve
    
    # Save clean image
    plt.figure(figsize=(3, 3))
    plt.imshow(clean, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('test_images/clean_tshirt.png')
    plt.close()
    
    # Generate adversarial version with subtle changes
    adversarial = clean.copy()
    # Add some subtle noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, (28, 28))
    adversarial = adversarial + noise
    # Add a targeted pattern
    adversarial[13:15, 13:15] = 0.9  # Small bright spot
    adversarial = np.clip(adversarial, 0, 1)
    
    # Save adversarial image
    plt.figure(figsize=(3, 3))
    plt.imshow(adversarial, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('test_images/adversarial_tshirt.png')
    plt.close()
    
    print("Generated sample images in test_images directory")
    
    # Create a comparison image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(clean, cmap='gray')
    plt.title("Clean Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(adversarial, cmap='gray')
    plt.title("Adversarial Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_images/comparison.png')
    plt.close()

# Create a README file for test images
def create_readme():
    readme_content = """# Test Images for Adversarial Attack Detection

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
"""
    
    with open('test_images/README.md', 'w') as f:
        f.write(readme_content)
    
    print("Created README file in test_images directory")

def main():
    print("Setting up sample images for adversarial attack detection testing...")
    
    # Create necessary directories
    create_directories()
    
    # Generate synthetic FMNIST-like images
    generate_sample_images()
    
    # Create README
    create_readme()
    
    print("\nSetup complete! Sample images are available in the test_images directory.")
    print("You can now run the detector with one of these images:")
    print("PYTHONPATH=. python3 src/detect_adversarial.py --image test_images/adversarial_tshirt.png")

if __name__ == "__main__":
    main() 