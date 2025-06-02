import os
import sys
import shutil
import subprocess
from pathlib import Path
import argparse

# Import our new visualization module
from visualize_detection import visualize_detection_advanced
from detect_adversarial import detect_adversarial_attacks
from models.cnn import TrainedModelProvider

def ensure_directory_exists(directory):
    """Ensure the specified directory exists"""
    os.makedirs(directory, exist_ok=True)
    print(f"Directory ready: {directory}")

def copy_image_to_project(source_path, destination_dir="my_images"):
    """Copy an image to the project directory"""
    # Ensure source file exists
    if not os.path.exists(source_path):
        print(f"Error: Source file not found: {source_path}")
        return None
        
    # Create destination directory if it doesn't exist
    ensure_directory_exists(destination_dir)
    
    # Get the filename from the source path
    filename = os.path.basename(source_path)
    
    # Create the destination path
    dest_path = os.path.join(destination_dir, filename)
    
    try:
        # Copy the file
        shutil.copy2(source_path, dest_path)
        print(f"Image successfully copied to: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error copying file: {e}")
        return None

def run_detection_with_visualization(image_path):
    """Run the adversarial attack detector with advanced visualization"""
    print(f"\nAnalyzing image for adversarial attacks: {image_path}")
    
    try:
        # Create results directory
        results_dir = os.path.join(os.getcwd(), 'docs')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate save path
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        save_path = os.path.join(results_dir, f"{name_without_ext}_detection.png")
        
        # Load model
        print("Loading model...")
        model_provider = TrainedModelProvider()
        model = model_provider.get_model()
        
        # Run detection
        print("Running detection...")
        results = detect_adversarial_attacks(image_path, model, verbose=True)
        
        # Generate visualization
        print("Generating visualization...")
        visualize_detection_advanced(image_path, results, save_path)
        
        # Print summary
        print("\nDetection Summary:")
        print(f"Class: {results['predicted_label']} (Confidence: {results['confidence']:.4f})")
        print(f"Risk Score: {results['risk_score']:.4f}")
        print(f"Assessment: {'LIKELY ADVERSARIAL' if results['is_adversarial'] else 'LIKELY CLEAN'}")
        print(f"Visualization saved to: {save_path}")
        
        return True
    except Exception as e:
        print(f"Error running detector: {e}")
        return False

def interactive_mode():
    """Interactive mode for when no arguments are provided"""
    print("\n===================================================")
    print("   Adversarial Attack Detector - Upload & Check")
    print("===================================================\n")
    print("This tool helps you analyze your own images for adversarial attacks.")
    print("You can provide the path to any image on your computer.")
    
    # Ask if the user wants to copy the image to the project
    copy_to_project = input("\nDo you want to save a copy of your image to the project? (y/n): ").lower() == 'y'
    
    # Get the image path
    print("\nPlease enter the path to your image file.")
    print("TIP: You can drag and drop the image from your file explorer into this terminal.")
    
    image_path = input("\nImage path: ").strip()
    
    if not image_path:
        print("No path entered. Exiting.")
        return
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found at: {image_path}")
        return
    
    print(f"\nImage selected: {image_path}")
    
    # Optionally copy to project
    if copy_to_project:
        dest_dir = input(f"\nEnter destination directory (default: my_images): ").strip() or "my_images"
        new_path = copy_image_to_project(image_path, dest_dir)
        if new_path:
            image_path = new_path
    
    # Run detection with new visualization
    run_detection_with_visualization(image_path)

def main():
    # Check if any arguments were provided
    if len(sys.argv) == 1:
        # No arguments provided, run in interactive mode
        interactive_mode()
        return
    
    # Regular argument parsing for command-line mode
    parser = argparse.ArgumentParser(description='Upload and analyze an image for adversarial attacks')
    parser.add_argument('image_path', type=str, help='Path to the image file to analyze')
    parser.add_argument('--copy', action='store_true', help='Copy the image to the project directory first')
    parser.add_argument('--dir', type=str, default='my_images', help='Directory to copy the image to')
    
    args = parser.parse_args()
    
    image_path = args.image_path
    
    # Check if the source file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
        
    print(f"Image selected: {image_path}")
    
    # Optionally copy to project
    if args.copy:
        new_path = copy_image_to_project(image_path, args.dir)
        if new_path:
            image_path = new_path
    
    # Run detection with new visualization
    run_detection_with_visualization(image_path)
    
if __name__ == "__main__":
    main() 