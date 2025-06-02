# Fashion MNIST Dataset Samples

This directory contains sample images from the Fashion MNIST dataset that you can use to test the adversarial attack detector.

## How to Use

### 1. Generate Dataset Samples

To generate new samples from the Fashion MNIST dataset:

```bash
./detect.sh --dataset --list 20
```

This will generate 20 random samples from the dataset and save them in this directory.

### 2. Analyze a Specific Sample

To analyze a specific sample for adversarial characteristics:

```bash
./detect.sh --dataset --sample 1
```

Replace `1` with the sample ID you want to analyze.

### 3. Generate and Analyze an Adversarial Version

To create an adversarial version of a sample and analyze both:

```bash
./detect.sh --dataset --sample 1 --adversarial
```

### 4. Interactive Dataset Browser

For an interactive experience, use:

```bash
./detect.sh --dataset
```

This will let you:
- Browse samples from the Fashion MNIST dataset
- Analyze specific samples for adversarial characteristics
- Generate adversarial versions of dataset images
- **Analyze your own images from any directory on your computer**

### 5. Analyze Your Own Images

The dataset browser now includes a dedicated option to analyze any image from any directory:

1. Run the dataset browser:
   ```bash
   ./detect.sh --dataset
   ```
   
2. Select option 5: "Analyze your own image from any directory"

3. When prompted, enter the full path to your image (or drag and drop the image into the terminal)

4. View the detection results to see if an adversarial attack was detected!

## Understanding the Results

The detector will analyze the images and tell you if they show signs of adversarial manipulation. The results include:

- Risk score (0-1 scale)
- Detection method scores
- Visual comparison of original and adversarial versions
- Clear indication if an adversarial attack was detected 