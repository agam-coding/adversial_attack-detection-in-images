import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from pathlib import Path

from detect_adversarial import preprocess_image, detect_adversarial_attacks, LABELS
from models.cnn import TrainedModelProvider
from scipy.ndimage import median_filter, gaussian_filter

def generate_perturbed_view(image):
    """Generate a visualization of potential perturbations in the image"""
    # Process the image
    processed_img = preprocess_image(image)
    
    # Apply median filter to estimate a "clean" version
    median_filtered = median_filter(processed_img, size=2)
    
    # Calculate perturbation as difference between original and filtered
    perturbation = processed_img - median_filtered
    
    # Enhance perturbation for visibility
    enhanced_perturbation = perturbation * 5
    
    # Create a visualization with the perturbation highlighted
    # Use red for positive perturbations, blue for negative
    rgb_perturbation = np.zeros((*processed_img.shape[:2], 3))
    
    # Red channel for positive perturbations
    rgb_perturbation[:, :, 0] = np.clip(enhanced_perturbation[:, :, 0] * 10, 0, 1)
    
    # Blue channel for negative perturbations
    rgb_perturbation[:, :, 2] = np.clip(-enhanced_perturbation[:, :, 0] * 10, 0, 1)
    
    # Add the original grayscale image as background (in green channel for contrast)
    rgb_perturbation[:, :, 1] = np.squeeze(processed_img) * 0.7
    
    return rgb_perturbation

def custom_visualization(image_path, results, save_path=None):
    """
    Create a custom visualization with:
    1. Original image
    2. Image with perturbations
    3. Detection scores
    """
    # Process the image
    processed_img = preprocess_image(image_path)
    
    # Generate perturbation view
    perturbed_view = generate_perturbed_view(image_path)
    
    # Set up a figure with custom layout
    fig = plt.figure(figsize=(12, 6))
    
    # Set up a red or green title based on detection results
    is_adversarial = results['is_adversarial']
    assessment = 'LIKELY ADVERSARIAL' if is_adversarial else 'LIKELY CLEAN'
    title_color = 'red' if is_adversarial else 'green'
    
    plt.suptitle(f"Adversarial Attack Detection: {assessment}", 
                fontsize=16, fontweight='bold', color=title_color, y=0.98)
    
    # Original image (left)
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
    ax1.imshow(np.squeeze(processed_img), cmap='gray')
    ax1.set_title(f"Original Image\nClassified as: {results['predicted_label']}", fontsize=12)
    ax1.axis('off')
    
    # Perturbation visualization (middle)
    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1)
    ax2.imshow(perturbed_view)
    ax2.set_title("Image with Perturbations\n(Red/Blue highlights show potential adversarial patterns)", fontsize=10)
    ax2.axis('off')
    
    # Detection scores (right)
    ax3 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
    
    # Get specific scores as requested
    key_metrics = {
        'Noise Reduction Test': results['detailed_scores'].get('noise_reduction_test', 0),
        'Perturbation Size': results['detailed_scores'].get('perturbation_size', 0),
        'Confidence Score': results['detailed_scores'].get('confidence_score', 0)
    }
    
    # Plot horizontal bar chart with key metrics
    metrics = list(key_metrics.keys())
    scores = list(key_metrics.values())
    
    y_pos = np.arange(len(metrics))
    bars = ax3.barh(y_pos, scores, align='center')
    
    # Color bars based on threshold
    for i, bar in enumerate(bars):
        if scores[i] > 0.5:
            bar.set_color('red')
        else:
            bar.set_color('skyblue')
    
    # Add text labels with exact values
    for i, v in enumerate(scores):
        ax3.text(v + 0.02, i, f"{v:.3f}", va='center', fontsize=10)
    
    # Add overall risk
    ax3.text(0.5, -0.2, f"Overall Risk Score: {results['risk_score']:.3f}", 
             ha='center', transform=ax3.transAxes, fontsize=12, 
             fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(metrics)
    ax3.set_xlim(0, 1.0)
    ax3.set_xlabel('Score (>0.5 is suspicious)')
    ax3.set_title("Detection Scores", fontsize=12)
    
    # Add threshold line
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Custom visualization saved to {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Custom adversarial attack visualization')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--save', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--dir', type=str, help='Process all images in directory')
    args = parser.parse_args()
    
    # Validate arguments - either image or dir must be provided
    if not args.image and not args.dir:
        print("Error: Either --image or --dir must be provided.")
        return
    
    # Load model
    print("Loading model...")
    model_provider = TrainedModelProvider()
    model = model_provider.get_model()
    
    if args.dir:
        # Process an entire directory
        process_directory(args.dir, model)
    else:
        # Process a single image
        process_single_image(args.image, args.save, model)

def process_single_image(image_path, save_path, model):
    """Process a single image and create visualization"""
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Detect adversarial attacks
    print(f"Analyzing image: {image_path}")
    results = detect_adversarial_attacks(image_path, model, verbose=True)
    
    # Determine save path
    if not save_path:
        # Create output directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'custom_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Use filename without extension
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        save_path = os.path.join(results_dir, f"{name_without_ext}_custom.png")
    
    # Create custom visualization
    fig = custom_visualization(image_path, results, save_path)
    
    # Print summary
    print("\nDetection Summary:")
    print(f"Class: {results['predicted_label']} (Confidence: {results['confidence']:.4f})")
    print(f"Risk Score: {results['risk_score']:.4f}")
    print(f"Assessment: {'LIKELY ADVERSARIAL' if results['is_adversarial'] else 'LIKELY CLEAN'}")
    print(f"Visualization saved to: {save_path}")
    
    plt.close(fig)

def process_directory(dir_path, model):
    """Process all images in a directory"""
    # Check if directory exists
    if not os.path.exists(dir_path):
        print(f"Error: Directory not found at {dir_path}")
        return
    
    # Create output directory
    results_dir = os.path.join(os.getcwd(), 'custom_results')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Scanning directory: {dir_path}")
    
    # Count processed images
    count = 0
    
    # Process each file in the directory
    for filename in os.listdir(dir_path):
        # Skip non-image files (simple check for common image extensions)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            continue
        
        # Full path to the image
        image_path = os.path.join(dir_path, filename)
        
        # Skip directories
        if os.path.isdir(image_path):
            continue
        
        # Create save path
        name_without_ext = os.path.splitext(filename)[0]
        save_path = os.path.join(results_dir, f"{name_without_ext}_custom.png")
        
        try:
            # Detect adversarial attacks
            print(f"\nAnalyzing image {count+1}: {filename}")
            results = detect_adversarial_attacks(image_path, model, verbose=False)
            
            # Create custom visualization
            fig = custom_visualization(image_path, results, save_path)
            plt.close(fig)
            
            # Print brief summary
            print(f"  Class: {results['predicted_label']} | Risk: {results['risk_score']:.4f}")
            print(f"  Assessment: {'LIKELY ADVERSARIAL' if results['is_adversarial'] else 'LIKELY CLEAN'}")
            print(f"  Visualization saved to: {save_path}")
            
            count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"\nProcessed {count} images. Results saved to {results_dir}/")

if __name__ == "__main__":
    main() 