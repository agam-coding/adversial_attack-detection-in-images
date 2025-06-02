# My Images Directory

This directory is for your own images that you want to analyze with the adversarial attack detector.

## How to Add Images Here

### Method 1: Manual Copy

Simply copy your image files to this directory.

### Method 2: Using the Upload Utility

```bash
./detect.sh --upload /path/to/your/image.jpg --copy
```

This command will automatically copy your image to this directory and run the analysis.

## Analyzing Images in This Directory

Once your images are here, you can analyze them with:

```bash
./detect.sh --image my_images/your_image.jpg
```

Or simply run the interactive mode and enter the path:

```bash
./detect.sh
```

Then at the prompt:
```
Enter path to image file (or 'q' to quit): my_images/your_image.jpg
``` 