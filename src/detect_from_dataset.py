import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import tensorflow as tf

from src.models.cnn import TrainedModelProvider
from src.datasets.fashion_mnist import FashionMnistDataset
from src.detect_adversarial import detect_adversarial_attacks, visualize_detection, LABELS

def list_dataset_samples(num_samples=10, save_dir='dataset_samples'):
    """List and save sample images from the Fashion MNIST dataset."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the dataset
    print("Loading Fashion MNIST dataset...")
    dataset = FashionMnistDataset()
    x_test, y_test = dataset.get_test()
    
    # Choose random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    # Create a montage of sample images
    ncols = min(5, num_samples)
    nrows = (num_samples + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()
    
    samples_info = []
    
    for i, idx in enumerate(indices):
        img = x_test[idx]
        label = np.argmax(y_test[idx])
        
        # Display image
        axes[i].imshow(np.squeeze(img), cmap='gray')
        axes[i].set_title(f"Sample {i+1}: {LABELS[label]}")
        axes[i].axis('off')
        
        # Save individual image
        sample_path = os.path.join(save_dir, f"sample_{i+1}.png")
        plt.figure(figsize=(3, 3))
        plt.imshow(np.squeeze(img), cmap='gray')
        plt.title(f"Sample {i+1}: {LABELS[label]}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(sample_path)
        plt.close()
        
        # Store info
        samples_info.append({
            'id': i+1,
            'index': int(idx),
            'label': LABELS[label],
            'class_id': int(label),
            'path': sample_path
        })
        
    plt.tight_layout()
    montage_path = os.path.join(save_dir, "samples_montage.png")
    plt.savefig(montage_path)
    plt.close()
    
    # Print information about the samples
    print(f"\nGenerated {num_samples} sample images from Fashion MNIST dataset")
    print(f"Samples saved to: {save_dir}/")
    print(f"Montage saved to: {montage_path}")
    print("\nAvailable samples:")
    
    for sample in samples_info:
        print(f"  Sample {sample['id']}: {sample['label']} (Class {sample['class_id']}) - {sample['path']}")
    
    return samples_info

def generate_adversarial_example(image, model, epsilon=0.1):
    """Generate an adversarial example using Fast Gradient Sign Method."""
    print("Generating adversarial example...")
    
    # Get the true label (guess based on model prediction)
    img_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0))
    prediction = model.predict(img_tensor, verbose=0)
    label = np.argmax(prediction[0])
    
    # Fast Gradient Sign Method
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy([label], prediction)
    
    # Get gradients
    gradient = tape.gradient(loss, img_tensor)
    
    # Create adversarial example
    signed_grad = tf.sign(gradient)
    adversarial = img_tensor + epsilon * signed_grad
    adversarial = tf.clip_by_value(adversarial, 0, 1)
    
    return adversarial.numpy()[0], int(label)

def analyze_dataset_image(index=None, sample_id=None, generate_adversarial=False, epsilon=0.1):
    """Analyze a specific image from the dataset."""
    # Load the dataset
    dataset = FashionMnistDataset()
    x_test, y_test = dataset.get_test()
    
    # Load model
    print("Loading model...")
    model_provider = TrainedModelProvider()
    model = model_provider.get_model()
    
    # Select the image
    if sample_id is not None:
        # Reload the samples info if we're using sample ID
        samples_info = list_dataset_samples(num_samples=10, save_dir='dataset_samples')
        if 1 <= sample_id <= len(samples_info):
            index = samples_info[sample_id-1]['index']
            print(f"Using sample {sample_id} (dataset index: {index})")
        else:
            print(f"Invalid sample ID. Please choose between 1 and {len(samples_info)}")
            return
    
    if index is None or index < 0 or index >= len(x_test):
        print(f"Invalid index. Please choose between 0 and {len(x_test)-1}")
        return
    
    # Get the image and true label
    image = x_test[index]
    true_label = np.argmax(y_test[index])
    
    if generate_adversarial:
        # Generate adversarial example
        adv_image, _ = generate_adversarial_example(image, model, epsilon)
        
        # Analyze both original and adversarial
        print("\n===== Analyzing Original Image =====")
        original_results = detect_adversarial_attacks(image, model)
        visualize_detection(image, original_results, 'docs/original_result.png')
        
        print("\n===== Analyzing Adversarial Version =====")
        adv_results = detect_adversarial_attacks(adv_image, model)
        visualize_detection(adv_image, adv_results, 'docs/adversarial_result.png')
        
        # Show comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(np.squeeze(image), cmap='gray')
        plt.title(f"Original: {LABELS[true_label]}\nRisk: {original_results['risk_score']:.4f}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(adv_image), cmap='gray')
        plt.title(f"Adversarial\nRisk: {adv_results['risk_score']:.4f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('docs/comparison_result.png')
        print(f"\nComparison saved to: docs/comparison_result.png")
        
    else:
        # Just analyze the original image
        print(f"\nAnalyzing dataset image index {index} (True label: {LABELS[true_label]})")
        results = detect_adversarial_attacks(image, model)
        visualize_detection(image, results, 'docs/dataset_result.png')

def interactive_mode():
    """Run interactive mode to browse and analyze dataset images."""
    # Load model once
    print("Loading model...")
    model_provider = TrainedModelProvider()
    model = model_provider.get_model()
    
    # Load dataset
    dataset = FashionMnistDataset()
    x_test, y_test = dataset.get_test()
    
    # Generate initial samples
    samples_info = list_dataset_samples(num_samples=10, save_dir='dataset_samples')
    
    print("\n" + "="*50)
    print("   Dataset Browser - Interactive Mode")
    print("="*50)
    print("\nThis tool lets you browse and analyze images from the Fashion MNIST dataset.")
    print("You can select specific images to check for adversarial characteristics.")
    print("You can also generate adversarial versions of any image.\n")
    
    while True:
        print("\nOptions:")
        print("  1. View more dataset samples")
        print("  2. Analyze a specific sample")
        print("  3. Analyze by dataset index")
        print("  4. Generate and analyze an adversarial example")
        print("  5. Analyze your own image from any directory")
        print("  6. Quit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            num = input("How many samples to generate? (default: 10): ")
            try:
                num = int(num) if num.strip() else 10
                samples_info = list_dataset_samples(num_samples=num, save_dir='dataset_samples')
            except ValueError:
                print("Invalid number. Using default of 10 samples.")
                samples_info = list_dataset_samples(num_samples=10, save_dir='dataset_samples')
                
        elif choice == '2':
            sample_id = input("Enter sample ID to analyze: ")
            try:
                sample_id = int(sample_id)
                analyze_dataset_image(sample_id=sample_id)
            except ValueError:
                print("Invalid sample ID. Please enter a number.")
        
        elif choice == '3':
            index = input(f"Enter dataset index (0-{len(x_test)-1}): ")
            try:
                index = int(index)
                analyze_dataset_image(index=index)
            except ValueError:
                print("Invalid index. Please enter a number.")
        
        elif choice == '4':
            source = input("Enter sample ID or dataset index? (s/i): ")
            if source.lower() == 's':
                id_input = input("Enter sample ID to generate adversarial version: ")
                try:
                    sample_id = int(id_input)
                    analyze_dataset_image(sample_id=sample_id, generate_adversarial=True)
                except ValueError:
                    print("Invalid sample ID. Please enter a number.")
            else:
                idx_input = input(f"Enter dataset index (0-{len(x_test)-1}): ")
                try:
                    index = int(idx_input)
                    analyze_dataset_image(index=index, generate_adversarial=True)
                except ValueError:
                    print("Invalid index. Please enter a number.")
        
        elif choice == '5':
            # Option to analyze a user's own image
            print("\n" + "="*50)
            print("  Analyze Your Own Image")
            print("="*50)
            print("\nYou can analyze any image from any directory on your computer.")
            print("TIP: You can drag and drop an image from your file explorer directly into this terminal.\n")
            
            image_path = input("Enter the full path to your image file: ")
            
            if not image_path.strip():
                print("No path entered. Returning to main menu.")
                continue
                
            if not os.path.exists(image_path):
                print(f"Error: Image file not found at '{image_path}'")
                print("Please provide the full path to the image file.")
                continue
            
            # Use the main detect_adversarial.py script to analyze the user's image
            print(f"\nAnalyzing your image: {image_path}")
            
            try:
                # Import the necessary function
                from src.detect_adversarial import detect_adversarial_attacks, preprocess_image
                
                # Process the image
                processed_img = preprocess_image(image_path)
                
                # Analyze the image
                results = detect_adversarial_attacks(processed_img, model)
                
                # Create a unique filename based on input image
                base_name = os.path.basename(image_path)
                file_name, ext = os.path.splitext(base_name)
                save_path = f'docs/user_image_{file_name}.png'
                
                # Visualize the results
                visualize_detection(processed_img, results, save_path)
                
                # Print clear conclusion
                print("\n" + "="*50)
                print("ADVERSARIAL ATTACK DETECTION RESULTS")
                print("="*50)
                print(f"Image analyzed: {image_path}")
                print(f"Predicted class: {results['predicted_label']} (Class {results['predicted_class']})")
                print(f"Confidence: {results['confidence']:.4f}")
                print(f"Risk score: {results['risk_score']:.4f} (Threshold: 0.5)")
                
                if results['is_adversarial']:
                    print("\n❌ ALERT: ADVERSARIAL ATTACK DETECTED")
                    print("This image shows signs of adversarial manipulation intended to fool neural networks.")
                else:
                    print("\n✅ NO ADVERSARIAL ATTACK DETECTED")
                    print("This image appears to be clean with no signs of adversarial manipulation.")
                
                print(f"\nDetailed visualization saved to: {save_path}")
                print("="*50)
                
            except Exception as e:
                print(f"\nError processing image: {e}")
                import traceback
                traceback.print_exc()
                print("\nPlease try another image.")
                
        elif choice == '6':
            print("\nExiting dataset browser. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

def main():
    parser = argparse.ArgumentParser(description='Browse and detect adversarial attacks in Fashion MNIST dataset')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--list', type=int, metavar='N', help='List N sample images from dataset')
    parser.add_argument('--index', type=int, help='Analyze specific dataset index')
    parser.add_argument('--sample', type=int, help='Analyze specific sample ID')
    parser.add_argument('--adversarial', action='store_true', help='Generate and analyze adversarial example')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for adversarial generation')
    
    args = parser.parse_args()
    
    # Ensure docs directory exists
    os.makedirs('docs', exist_ok=True)
    
    if args.interactive:
        interactive_mode()
    elif args.list:
        list_dataset_samples(num_samples=args.list)
    elif args.index is not None:
        analyze_dataset_image(index=args.index, generate_adversarial=args.adversarial, epsilon=args.epsilon)
    elif args.sample is not None:
        analyze_dataset_image(sample_id=args.sample, generate_adversarial=args.adversarial, epsilon=args.epsilon)
    else:
        # Default to interactive mode
        interactive_mode()

if __name__ == "__main__":
    main() 