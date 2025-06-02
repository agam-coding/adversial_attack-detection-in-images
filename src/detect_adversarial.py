import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from PIL import Image

from src.models.cnn import TrainedModelProvider
from src.datasets.fashion_mnist import FashionMnistDataset
from src.datasets.reduced import ReducedDataset
from src.experiments.experiment import Experiment, GAParameters
from src.experiments.ga_functions import compute_perturbation_size


# FMNIST class labels
LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Set of detection methods to use
DETECTION_METHODS = [
    "confidence_score",
    "prediction_consistency",
    "feature_squeezing",
    "perturbation_size",
    "noise_reduction_test"
]


def preprocess_image(image_path, target_size=(28, 28)):
    """Load and preprocess a single image."""
    if isinstance(image_path, str) and os.path.exists(image_path):
        # Load external image file
        try:
            # Try loading with keras first
            img = load_img(image_path, color_mode='grayscale', target_size=target_size)
            img_array = img_to_array(img)
            img_array = img_array.astype('float32') / 255.0
            return img_array
        except Exception as e:
            print(f"Error loading image with keras: {e}")
            print("Trying alternate loading method...")
            try:
                # Try with PIL directly
                img = Image.open(image_path)
                
                # Convert to grayscale if not already
                if img.mode != 'L':
                    print(f"Converting {img.mode} image to grayscale")
                    img = img.convert('L')
                
                # Resize to target size
                img = img.resize(target_size)
                
                # Convert to numpy array and normalize
                img_array = np.array(img).reshape(target_size[0], target_size[1], 1)
                img_array = img_array.astype('float32') / 255.0
                return img_array
            except Exception as e2:
                print(f"Error loading image with PIL: {e2}")
                raise ValueError(f"Could not load image: {image_path}")
    else:
        # If not a valid path, assume it's a numpy array already
        img_array = image_path
        
        # If array has wrong dimensions, try to reshape
        if len(img_array.shape) > 2:
            # Handle RGB images by converting to grayscale
            if img_array.shape[-1] == 3 or img_array.shape[-1] == 4:
                print("Converting color image to grayscale")
                # Simple grayscale conversion using mean across channels
                img_array = np.mean(img_array, axis=-1, keepdims=True)
            
        # Ensure correct shape
        if img_array.shape[:2] != target_size:
            print(f"Resizing image from {img_array.shape[:2]} to {target_size}")
            # Resize using nearest neighbor
            from skimage.transform import resize
            img_array = resize(img_array, (target_size[0], target_size[1], 1), 
                              anti_aliasing=True, preserve_range=True)
        
        # Ensure values are in [0,1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
            
        return img_array


def detect_adversarial_attacks(image, model, detection_methods=None, verbose=True, 
                              confidence_threshold=0.8, perturbation_threshold=0.05,
                              consistency_threshold=0.2):
    """
    Detect if an image is an adversarial example using multiple detection methods.
    
    Args:
        image: The input image, path or numpy array
        model: The classifier model to test against
        detection_methods: List of detection methods to use
        verbose: Whether to print detailed results
        confidence_threshold: Threshold for confidence score
        perturbation_threshold: Threshold for perturbation size
        consistency_threshold: Threshold for prediction consistency
        
    Returns:
        Dictionary with detection results and risk score
    """
    if detection_methods is None:
        detection_methods = DETECTION_METHODS
        
    # Preprocess the image
    processed_img = preprocess_image(image)
    if processed_img.shape != (28, 28, 1):
        processed_img = processed_img.reshape(28, 28, 1)
    
    # Create a batch of 1
    img_batch = np.expand_dims(processed_img, axis=0)
    
    # Get predictions
    predictions = model.predict(img_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Initialize results
    results = {
        "predicted_class": predicted_class,
        "predicted_label": LABELS[predicted_class],
        "confidence": float(confidence),
        "detection_results": {},
        "is_adversarial": False,
        "risk_score": 0.0,
        "detailed_scores": {}
    }
    
    # Method 1: Confidence Score - adversarial examples often have lower confidence
    if "confidence_score" in detection_methods:
        confidence_result = confidence < confidence_threshold
        confidence_risk = 1.0 - float(confidence)
        results["detection_results"]["confidence_score"] = confidence_result
        results["detailed_scores"]["confidence_score"] = confidence_risk
    
    # Method 2: Prediction Consistency - run with small noise to see if prediction changes
    if "prediction_consistency" in detection_methods:
        # Add small random noise
        noisy_img = img_batch + np.random.normal(0, 0.01, img_batch.shape)
        noisy_img = np.clip(noisy_img, 0, 1)
        
        # Get predictions with noise
        noisy_predictions = model.predict(noisy_img, verbose=0)
        noisy_class = np.argmax(noisy_predictions[0])
        
        # Check if prediction changes with small noise
        consistency_result = predicted_class != noisy_class
        
        # Calculate consistency score (L1 distance between prediction vectors)
        consistency_score = np.mean(np.abs(predictions - noisy_predictions))
        results["detection_results"]["prediction_consistency"] = consistency_score > consistency_threshold
        results["detailed_scores"]["prediction_consistency"] = float(consistency_score)
    
    # Method 3: Feature Squeezing - reduce color depth and check if prediction changes
    if "feature_squeezing" in detection_methods:
        # Simulate bit depth reduction (e.g., to 5 bits)
        bit_depth = 5
        max_val = 2**bit_depth - 1
        squeezed_img = np.round(img_batch * max_val) / max_val
        
        # Get predictions with squeezed features
        squeezed_predictions = model.predict(squeezed_img, verbose=0)
        squeezed_class = np.argmax(squeezed_predictions[0])
        
        # Check if prediction changes after squeezing
        squeezing_result = predicted_class != squeezed_class
        
        # Calculate score (L1 distance between prediction vectors)
        squeezing_score = np.mean(np.abs(predictions - squeezed_predictions))
        results["detection_results"]["feature_squeezing"] = squeezing_result
        results["detailed_scores"]["feature_squeezing"] = float(squeezing_score)
    
    # Method 4: Perturbation Size Estimation
    if "perturbation_size" in detection_methods:
        # Get a "clean" reference (we don't have actual clean image)
        # So we apply median filter to estimate the clean version
        from scipy.ndimage import median_filter
        median_filtered = median_filter(img_batch[0], size=2).reshape(img_batch.shape)
        
        # Estimate perturbation as difference between original and filtered
        perturbation = img_batch - median_filtered
        
        # Calculate perturbation size using project's function
        flat_perturbation = perturbation.reshape(28, 28)
        pert_size = compute_perturbation_size(flat_perturbation, pm1=10, pm2=5.8)
        
        # Normalize to 0-1 range
        normalized_size = min(pert_size / 10.0, 1.0)  # Assuming max reasonable size is around 10
        perturbation_result = normalized_size > perturbation_threshold
        results["detection_results"]["perturbation_size"] = perturbation_result
        results["detailed_scores"]["perturbation_size"] = float(normalized_size)
    
    # Method 5: Noise Reduction Test
    if "noise_reduction_test" in detection_methods:
        # Apply Gaussian blur to reduce noise
        from scipy.ndimage import gaussian_filter
        smoothed_img = gaussian_filter(img_batch[0], sigma=0.5).reshape(img_batch.shape)
        
        # Get predictions with smoothed image
        smoothed_predictions = model.predict(smoothed_img, verbose=0)
        smoothed_class = np.argmax(smoothed_predictions[0])
        
        # Check if prediction changes after smoothing
        smoothing_result = predicted_class != smoothed_class
        
        # Calculate score (L1 distance)
        smoothing_score = np.mean(np.abs(predictions - smoothed_predictions))
        results["detection_results"]["noise_reduction_test"] = smoothing_result
        results["detailed_scores"]["noise_reduction_test"] = float(smoothing_score)
    
    # Calculate overall risk score (weighted average of all scores)
    weights = {
        "confidence_score": 0.3,
        "prediction_consistency": 0.2,
        "feature_squeezing": 0.15,
        "perturbation_size": 0.25,
        "noise_reduction_test": 0.1
    }
    
    risk_score = 0.0
    weight_sum = 0.0
    
    for method, score in results["detailed_scores"].items():
        if method in weights:
            risk_score += score * weights[method]
            weight_sum += weights[method]
    
    if weight_sum > 0:
        risk_score /= weight_sum
    
    # Determine if adversarial based on risk score
    results["risk_score"] = float(risk_score)
    results["is_adversarial"] = risk_score > 0.5
    
    # Print results if verbose
    if verbose:
        print(f"Prediction: {results['predicted_label']} (Class {results['predicted_class']})")
        print(f"Confidence: {results['confidence']:.4f}")
        print("\nDetection Results:")
        
        for method, result in results["detection_results"].items():
            print(f"  {method}: {'SUSPICIOUS' if result else 'CLEAN'} " +
                  f"(Score: {results['detailed_scores'].get(method, 0):.4f})")
        
        print(f"\nRisk Score: {results['risk_score']:.4f}")
        print(f"Overall Assessment: {'LIKELY ADVERSARIAL' if results['is_adversarial'] else 'LIKELY CLEAN'}")
    
    return results


def visualize_detection(image, results, save_path=None):
    """Visualize the detection results with the image."""
    processed_img = preprocess_image(image)
    
    # Get the original image for display if it's a file path
    original_img = None
    if isinstance(image, str) and os.path.exists(image):
        try:
            from PIL import Image
            original_img = np.array(Image.open(image))
        except:
            original_img = None
    
    # Set up the figure - wider if we have the original image
    if original_img is not None:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot the original image
        if len(original_img.shape) == 3 and original_img.shape[2] == 3:
            ax0.imshow(original_img)
        else:
            ax0.imshow(original_img, cmap='gray')
        ax0.set_title("Original Image")
        ax0.axis('off')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the processed image
    ax1.imshow(np.squeeze(processed_img), cmap='gray')
    ax1.set_title(f"Classified as: {results['predicted_label']}\n" +
                  f"Confidence: {results['confidence']:.4f}")
    ax1.axis('off')
    
    # Plot the risk scores
    methods = list(results["detailed_scores"].keys())
    scores = [results["detailed_scores"][m] for m in methods]
    
    # Improve method names for display
    method_names = [' '.join(m.split('_')).title() for m in methods]
    
    # Plot bar chart of detection scores
    bars = ax2.barh(method_names, scores, color='skyblue')
    
    # Color bars based on threshold
    for i, bar in enumerate(bars):
        if scores[i] > 0.5:
            bar.set_color('salmon')
    
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_title(f"Detection Scores\nOverall Risk: {results['risk_score']:.4f}")
    ax2.set_xlabel('Risk Score (>0.5 is suspicious)')
    
    # Add overall assessment
    assessment = 'LIKELY ADVERSARIAL' if results['is_adversarial'] else 'LIKELY CLEAN'
    fig.suptitle(f"Adversarial Attack Detection: {assessment}", 
                fontsize=16, fontweight='bold', 
                color='red' if results['is_adversarial'] else 'green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.close()  # Close the figure to avoid displaying in notebooks


def test_fmnist_dataset(model, num_samples=10):
    """Test detection on clean FMNIST examples."""
    dataset = FashionMnistDataset()
    x_test, y_test = dataset.get_test()
    
    # Randomly select samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    clean_scores = []
    for idx in indices:
        img = x_test[idx]
        results = detect_adversarial_attacks(img, model, verbose=False)
        clean_scores.append(results["risk_score"])
        print(f"Clean sample {idx}: Risk score = {results['risk_score']:.4f}, " +
             f"Actual class: {LABELS[np.argmax(y_test[idx])]}, " +
             f"Predicted: {results['predicted_label']}")
    
    return clean_scores


def generate_adversarial_examples(model, dataset, num_samples=10, epsilon=0.1):
    """Generate simple adversarial examples using Fast Gradient Sign Method (FGSM)."""
    x_test, y_test = dataset.get_test()
    
    # Randomly select samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    adversarial_examples = []
    original_images = []
    
    for idx in indices:
        img = x_test[idx]
        label = np.argmax(y_test[idx])
        
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(np.expand_dims(img, axis=0))
        label_tensor = tf.convert_to_tensor([label])
        
        # Fast Gradient Sign Method
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            prediction = model(img_tensor)
            loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
        
        # Get gradients
        gradient = tape.gradient(loss, img_tensor)
        
        # Create adversarial example
        signed_grad = tf.sign(gradient)
        adversarial = img_tensor + epsilon * signed_grad
        adversarial = tf.clip_by_value(adversarial, 0, 1)
        
        adversarial_examples.append(adversarial.numpy()[0])
        original_images.append(img)
    
    return original_images, adversarial_examples, [np.argmax(y_test[i]) for i in indices]


def test_adversarial_examples(model, adversarials, original_labels):
    """Test detection on adversarial examples."""
    adversarial_scores = []
    
    for i, (adv, true_label) in enumerate(zip(adversarials, original_labels)):
        results = detect_adversarial_attacks(adv, model, verbose=False)
        adversarial_scores.append(results["risk_score"])
        print(f"Adversarial sample {i}: Risk score = {results['risk_score']:.4f}, " +
             f"True class: {LABELS[true_label]}, " +
             f"Predicted: {results['predicted_label']}")
    
    return adversarial_scores


def evaluate_detector_performance(clean_scores, adv_scores):
    """Evaluate detector performance with ROC curve and precision-recall curve."""
    # Combine scores and create labels (0=clean, 1=adversarial)
    all_scores = np.concatenate([clean_scores, adv_scores])
    all_labels = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(adv_scores))])
    
    # Calculate ROC curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve and average precision
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    
    # Plot ROC curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    # Plot precision-recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig('docs/detector_performance.png')
    plt.show()
    
    return {
        'roc_auc': roc_auc,
        'average_precision': ap
    }


def evaluate_detector(model, num_samples=10, epsilon=0.1):
    """Evaluate the detector on both clean and adversarial examples."""
    dataset = FashionMnistDataset()
    
    print("\n======= Testing on Clean Examples =======")
    clean_scores = test_fmnist_dataset(model, num_samples)
    
    print("\n======= Generating Adversarial Examples =======")
    original_images, adversarial_examples, original_labels = generate_adversarial_examples(
        model, dataset, num_samples, epsilon)
    
    print("\n======= Testing on Adversarial Examples =======")
    adversarial_scores = test_adversarial_examples(model, adversarial_examples, original_labels)
    
    print("\n======= Detector Performance Evaluation =======")
    metrics = evaluate_detector_performance(clean_scores, adversarial_scores)
    
    print(f"\nROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    # Show a few examples
    for i in range(min(3, num_samples)):
        # Clean example
        clean_results = detect_adversarial_attacks(original_images[i], model, verbose=False)
        visualize_detection(original_images[i], clean_results, f'docs/clean_example_{i}.png')
        
        # Adversarial example
        adv_results = detect_adversarial_attacks(adversarial_examples[i], model, verbose=False)
        visualize_detection(adversarial_examples[i], adv_results, f'docs/adversarial_example_{i}.png')


def main():
    parser = argparse.ArgumentParser(description='Detect adversarial attacks in images')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Run evaluation on test data instead of a single image')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples for evaluation')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Perturbation size for adversarial examples')
    parser.add_argument('--save', type=str, help='Path to save visualization')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model_provider = TrainedModelProvider()
    model = model_provider.get_model()
    
    if args.evaluate:
        # Run evaluation on test data
        evaluate_detector(model, args.samples, args.epsilon)
    elif args.image:
        # Run detection on a single image
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at {args.image}")
            return
        
        # Detect adversarial attack
        print(f"Analyzing image: {args.image}")
        results = detect_adversarial_attacks(args.image, model)
        
        # Visualize results
        save_path = args.save if args.save else 'docs/detection_result.png'
        visualize_detection(args.image, results, save_path)
        
        # Print a clear conclusion
        print("\n" + "="*50)
        print("ADVERSARIAL ATTACK DETECTION RESULTS")
        print("="*50)
        print(f"Image analyzed: {args.image}")
        print(f"Prediction: {results['predicted_label']} (Class {results['predicted_class']})")
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
    else:
        # Start interactive mode
        print("\n" + "="*50)
        print("   Adversarial Attack Detector - Interactive Mode")
        print("="*50)
        print("\nThis tool analyzes images to detect potential adversarial attacks.")
        print("It works best with Fashion MNIST style images (grayscale, 28x28), but will")
        print("attempt to process any image file by converting to grayscale and resizing.")
        print("\nSupported image formats: jpg, jpeg, png, bmp, and others.")
        print("\nDetection Methods:")
        for method in DETECTION_METHODS:
            print(f"  - {' '.join(method.split('_')).title()}")
        
        print("\nEach image will be analyzed and results will be saved to the 'docs' directory.")
        print("\n" + "-"*50)
        
        while True:
            try:
                image_path = input("\nEnter path to image file (or 'q' to quit): ")
                
                if image_path.lower() in ['q', 'quit', 'exit']:
                    print("\nExiting. Goodbye!")
                    break
                
                if not image_path.strip():
                    print("Please enter a valid path.")
                    continue
                
                if not os.path.exists(image_path):
                    print(f"Error: Image file not found at '{image_path}'")
                    print("Please provide the full path to the image file.")
                    continue
                
                # Detect adversarial attack
                print(f"\nAnalyzing image: {image_path}")
                print("Processing...")
                results = detect_adversarial_attacks(image_path, model)
                
                # Create a unique filename based on input image
                base_name = os.path.basename(image_path)
                file_name, ext = os.path.splitext(base_name)
                save_path = f'docs/detection_{file_name}_{Path().resolve().stem}.png'
                
                # Visualize results
                visualize_detection(image_path, results, save_path)
                
                # Print summary with a clearer assessment
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


if __name__ == "__main__":
    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)
    main() 