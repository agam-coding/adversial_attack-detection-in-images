import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from pathlib import Path

# Fix imports to avoid ModuleNotFoundError
from detect_adversarial import preprocess_image, detect_adversarial_attacks, LABELS
from models.cnn import TrainedModelProvider

def visualize_detection_advanced(image_path, results, save_path=None):
    """
    Create an advanced visualization of adversarial attack detection results,
    similar to the format shown in the sample image.
    """
    # Process the image
    processed_img = preprocess_image(image_path)
    
    # Create figure with a specific layout
    fig = plt.figure(figsize=(15, 7))
    
    # Set up a red or green title based on detection results
    is_adversarial = results['is_adversarial']
    assessment = 'LIKELY ADVERSARIAL' if is_adversarial else 'LIKELY CLEAN'
    title_color = 'red' if is_adversarial else 'green'
    
    plt.suptitle(f"Adversarial Attack Detection: {assessment}", 
                 fontsize=16, fontweight='bold', color=title_color, y=0.98)
    
    # Original image on the left (1/3 of width)
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
    ax1.imshow(np.squeeze(processed_img), cmap='gray')
    ax1.set_title(f"Sample 1: {results['predicted_label']}", fontsize=14)
    ax1.axis('off')
    
    # Processed/classified image in the middle (1/3 of width)
    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1)
    ax2.imshow(np.squeeze(processed_img), cmap='gray')
    ax2.set_title(f"Classified as: {results['predicted_label']}\nConfidence: {results['confidence']:.4f}", fontsize=12)
    ax2.axis('off')
    
    # Add classification info in text box at the top of middle section
    text_box = plt.figtext(0.5, 0.9, f"Confidence: {results['confidence']:.4f}", 
                           ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Detection scores bar chart on the right (1/3 of width)
    ax3 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
    
    # Get the method names and scores
    methods = list(results["detailed_scores"].keys())
    scores = [results["detailed_scores"][m] for m in methods]
    
    # Improve method names for display
    method_names = [' '.join(m.split('_')).title() for m in methods]
    
    # Plot bar chart of detection scores - horizontally
    y_pos = np.arange(len(method_names))
    bars = ax3.barh(y_pos, scores, align='center', color='skyblue')
    
    # Color bars based on threshold
    for i, bar in enumerate(bars):
        if scores[i] > 0.5:
            bar.set_color('salmon')
    
    # Set labels and title
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(method_names)
    ax3.invert_yaxis()  # labels read top-to-bottom
    ax3.set_xlabel('Risk Score (>0.5 is suspicious)')
    ax3.set_title(f"Detection Scores\nOverall Risk: {results['risk_score']:.4f}", fontsize=12)
    
    # Add threshold line
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlim(0, 1.0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the title
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize adversarial attack detection')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--save', type=str, default=None, help='Path to save visualization')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return
    
    # Load model
    print("Loading model...")
    model_provider = TrainedModelProvider()
    model = model_provider.get_model()
    
    # Detect adversarial attacks
    print(f"Analyzing image: {args.image}")
    results = detect_adversarial_attacks(args.image, model, verbose=True)
    
    # Determine save path
    save_path = args.save
    if not save_path:
        # Create 'results' directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'docs')
        os.makedirs(results_dir, exist_ok=True)
        
        # Use filename without extension + _detection.png
        base_name = os.path.basename(args.image)
        name_without_ext = os.path.splitext(base_name)[0]
        save_path = os.path.join(results_dir, f"{name_without_ext}_detection.png")
    
    # Visualize detection results
    fig = visualize_detection_advanced(args.image, results, save_path)
    
    # Print summary
    print("\nDetection Summary:")
    print(f"Class: {results['predicted_label']} (Confidence: {results['confidence']:.4f})")
    print(f"Risk Score: {results['risk_score']:.4f}")
    print(f"Assessment: {'LIKELY ADVERSARIAL' if results['is_adversarial'] else 'LIKELY CLEAN'}")
    print(f"Visualization saved to: {save_path}")
    
    plt.close(fig)

if __name__ == "__main__":
    main() 