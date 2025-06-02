import os
import sys
import shutil
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import local modules
sys.path.append(".")
from detect_adversarial import preprocess_image, detect_adversarial_attacks, LABELS
from models.cnn import TrainedModelProvider, CnnModelProvider
from datasets.fashion_mnist import FashionMnistDataset
from scipy.ndimage import median_filter, gaussian_filter

# Set page config
st.set_page_config(
    page_title="Adversarial Attack Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #26A69A;
        margin-bottom: 20px;
    }
    .success-text {
        color: green;
        font-weight: bold;
    }
    .warning-text {
        color: red;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

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
    
    return rgb_perturbation, processed_img

def custom_visualization(image_path, results):
    """Create custom visualization for Streamlit display"""
    # Generate perturbation view
    perturbed_view, processed_img = generate_perturbed_view(image_path)
    
    # Get specific scores as requested
    key_metrics = {
        'Noise Reduction Test': results['detailed_scores'].get('noise_reduction_test', 0),
        'Perturbation Size': results['detailed_scores'].get('perturbation_size', 0),
        'Confidence Score': results['detailed_scores'].get('confidence_score', 0)
    }
    
    return processed_img, perturbed_view, key_metrics, results

def detect_image(image_path):
    """Run detection on an image and return visualization components"""
    # Load model
    model_provider = TrainedModelProvider()
    model = model_provider.get_model()
    
    # Detect adversarial attacks
    results = detect_adversarial_attacks(image_path, model, verbose=False)
    
    # Create visualization
    processed_img, perturbed_view, key_metrics, results = custom_visualization(image_path, results)
    
    return processed_img, perturbed_view, key_metrics, results

class Trainer:
    def __init__(self, model_provider, dataset):
        self.model = model_provider.get_model()
        self.dataset = dataset
        self.training_progress = []

    def train(self, batch_size=32, epochs=20, use_augmentation=False):
        x_train, y_train = self.dataset.get_train()
        x_valid, y_valid = self.dataset.get_valid()

        self.model.compile(loss=CategoricalCrossentropy(),
                           optimizer=Adam(),
                           metrics=[CategoricalAccuracy()])
        
        # Use a custom callback to track progress for Streamlit
        class TrainingCallback(tf.keras.callbacks.Callback):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer
            
            def on_epoch_end(self, epoch, logs=None):
                self.trainer.training_progress.append(logs)
        
        callbacks = [
            EarlyStopping(patience=2, restore_best_weights=True),
            TrainingCallback(self)
        ]

        if use_augmentation:
            datagen = ImageDataGenerator(width_shift_range=0.05, height_shift_range=0.05,
                                         rotation_range=10, shear_range=15, zoom_range=[0.9, 1.1])
            datagen.fit(x_train[..., np.newaxis])

            history = self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                           steps_per_epoch=len(x_train) // batch_size,
                           epochs=epochs,
                           validation_data=(x_valid, y_valid),
                           verbose=1,
                           callbacks=callbacks)
        else:
            history = self.model.fit(x_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(x_valid, y_valid),
                           verbose=1,
                           callbacks=callbacks)
        
        return history

    def evaluate(self):
        x, y = self.dataset.get_test()
        score = self.model.evaluate(x, y, verbose=0)
        return score[1]  # Return accuracy

    def save(self, path='./weights/classifier'):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

def detect_section():
    st.markdown('<div class="sub-header">Upload Image for Adversarial Attack Detection</div>', unsafe_allow_html=True)
    
    st.info("Upload an image to analyze it for signs of adversarial attacks. The detector will check for perturbations that might fool machine learning models.")
    
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "bmp"])
    
    if uploaded_file is not None:
        # Create temp directory if it doesn't exist
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        
        # Show processing spinner
        with st.spinner("Analyzing image for adversarial attacks..."):
            # Perform detection
            processed_img, perturbed_view, key_metrics, results = detect_image(file_path)
        
        # Display results in three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Image")
            st.image(processed_img, use_column_width=True)
            st.text(f"Classified as: {results['predicted_label']}")
            st.text(f"Confidence: {results['confidence']:.4f}")
        
        with col2:
            st.subheader("Perturbation Visualization")
            st.image(perturbed_view, use_column_width=True)
            st.text("Red/blue highlights show potential adversarial patterns")
        
        with col3:
            st.subheader("Detection Scores")
            
            # Display metrics as a bar chart
            metrics = list(key_metrics.keys())
            scores = list(key_metrics.values())
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(metrics, scores, color=['skyblue', 'skyblue', 'skyblue'])
            
            # Color bars based on threshold
            for i, bar in enumerate(bars):
                if scores[i] > 0.5:
                    bar.set_color('red')
            
            # Add text labels with exact values
            for i, v in enumerate(scores):
                ax.text(v + 0.02, i, f"{v:.3f}", va='center')
            
            # Add overall risk
            ax.text(0.5, -0.2, f"Overall Risk Score: {results['risk_score']:.3f}", 
                    ha='center', transform=ax.transAxes, fontsize=12, 
                    fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_xlim(0, 1.0)
            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
        
        # Overall assessment
        if results['is_adversarial']:
            st.markdown(f"<h2 class='warning-text'>Assessment: LIKELY ADVERSARIAL</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 class='success-text'>Assessment: LIKELY CLEAN</h2>", unsafe_allow_html=True)

def train_section():
    st.markdown('<div class="sub-header">Train Adversarial Detection Model</div>', unsafe_allow_html=True)
    
    st.info("Train a model on the Fashion MNIST dataset. You can configure training parameters and use data augmentation.")
    
    # Training configuration
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs", min_value=1, max_value=50, value=10)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    with col2:
        use_augmentation = st.checkbox("Use Data Augmentation", value=True)
        save_model = st.checkbox("Save Model After Training", value=True)
    
    # Training button
    if st.button("Start Training"):
        with st.spinner("Initializing training..."):
            model_provider = CnnModelProvider()
            dataset = FashionMnistDataset()
            trainer = Trainer(model_provider, dataset)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create placeholder for metrics
        train_metrics = st.empty()
        
        # Start training
        with st.spinner("Training model..."):
            status_text.text("Training in progress...")
            history = trainer.train(batch_size=batch_size, epochs=epochs, use_augmentation=use_augmentation)
            
            # Update progress during training
            for i, logs in enumerate(trainer.training_progress):
                # Update progress bar
                progress = (i + 1) / epochs
                progress_bar.progress(progress)
                
                # Display metrics
                metrics_text = f"Epoch {i+1}/{epochs} - "
                metrics_text += f"Loss: {logs.get('loss', 0):.4f} - "
                metrics_text += f"Accuracy: {logs.get('categorical_accuracy', 0):.4f} - "
                metrics_text += f"Val Loss: {logs.get('val_loss', 0):.4f} - "
                metrics_text += f"Val Accuracy: {logs.get('val_categorical_accuracy', 0):.4f}"
                
                status_text.text(metrics_text)
                
                # Plot metrics
                if i > 0:
                    epochs_range = range(1, i+2)
                    acc = [logs.get('categorical_accuracy', 0) for logs in trainer.training_progress]
                    val_acc = [logs.get('val_categorical_accuracy', 0) for logs in trainer.training_progress]
                    loss = [logs.get('loss', 0) for logs in trainer.training_progress]
                    val_loss = [logs.get('val_loss', 0) for logs in trainer.training_progress]
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(epochs_range, acc, label='Training Accuracy')
                    ax1.plot(epochs_range, val_acc, label='Validation Accuracy')
                    ax1.set_title('Training and Validation Accuracy')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Accuracy')
                    ax1.legend()
                    
                    ax2.plot(epochs_range, loss, label='Training Loss')
                    ax2.plot(epochs_range, val_loss, label='Validation Loss')
                    ax2.set_title('Training and Validation Loss')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    
                    plt.tight_layout()
                    train_metrics.pyplot(fig)
                    plt.close(fig)
        
        # Evaluate model
        with st.spinner("Evaluating model..."):
            test_accuracy = trainer.evaluate()
        
        # Save model if requested
        if save_model:
            with st.spinner("Saving model..."):
                trainer.save()
            st.success("Model saved successfully!")
        
        # Display final results
        st.success(f"Training completed! Test accuracy: {test_accuracy:.4f}")

def main():
    st.markdown('<h1 class="main-header">Adversarial Attack Detector</h1>', unsafe_allow_html=True)
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Detect Adversarial Attacks", "Train Model"])
    
    # Display app information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("## About")
    st.sidebar.info(
        "This application helps detect adversarial attacks on images "
        "and allows training of custom detection models. "
        "Upload an image to check for potential adversarial perturbations."
    )
    
    # Main content based on selected page
    if page == "Detect Adversarial Attacks":
        detect_section()
    else:
        train_section()

if __name__ == "__main__":
    main()