"""
Make predictions on new MNIST samples using the trained model
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import sys
import argparse
from PIL import Image


def load_model(model_path='mnist_model.h5'):
    """Load the trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model


def preprocess_image(image):
    """
    Preprocess a single image for prediction
    Image should be 28x28 grayscale, values 0-255
    """
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure it's grayscale (2D)
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            # Convert RGB to grayscale
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        elif image.shape[2] == 4:
            # Convert RGBA to grayscale
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            image = image[:, :, 0]
    
    # Resize to 28x28 if needed
    if image.shape != (28, 28):
        img_pil = Image.fromarray(image.astype('uint8'))
        # Use LANCZOS resampling (compatible with older Pillow versions)
        try:
            img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        except AttributeError:
            # Fallback for older Pillow versions
            img_pil = img_pil.resize((28, 28), Image.LANCZOS)
        image = np.array(img_pil)
    
    # Normalize to [0, 1]
    image = image.astype('float32') / 255.0
    
    # Flatten to (784,)
    image = image.reshape(784)
    
    return image


def load_image_from_file(file_path):
    """Load and preprocess an image from file"""
    try:
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        return preprocess_image(image)
    except Exception as e:
        raise ValueError(f"Error loading image from {file_path}: {str(e)}")


def predict_single(model, image):
    """Make prediction on a single image"""
    # Reshape for batch prediction (1, 784)
    image_batch = image.reshape(1, 784)
    
    # Make prediction
    predictions = model.predict(image_batch, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit]
    
    return predicted_digit, confidence, predictions[0]


def predict_batch(model, images):
    """Make predictions on a batch of images"""
    # Ensure images are in correct shape (batch_size, 784)
    images = np.array(images)  # Ensure it's a numpy array
    
    if len(images.shape) == 1:
        # Single image: (784,) -> (1, 784)
        images = images.reshape(1, 784)
    elif len(images.shape) == 2:
        # Should be (N, 784) already, but verify
        if images.shape[1] != 784:
            # Flatten if needed (e.g., if it's (N, 28, 28))
            images = images.reshape(images.shape[0], -1)
    
    # Verify final shape
    if len(images.shape) != 2 or images.shape[1] != 784:
        raise ValueError(f"Expected images shape (N, 784), got {images.shape}")
    
    # Make predictions
    predictions = model.predict(images, verbose=0)
    predicted_digits = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    return predicted_digits, confidences, predictions


def predict_from_test_set(model, num_samples=10):
    """Make predictions on random samples from MNIST test set"""
    print(f"\nLoading {num_samples} random samples from MNIST test set...")
    
    # Load test data
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    
    # Get random samples
    indices = np.random.choice(x_test.shape[0], num_samples, replace=False)
    samples = x_test[indices]
    true_labels = y_test[indices]
    
    # Make predictions
    predicted_digits, confidences, all_predictions = predict_batch(model, samples)
    
    # Display results
    print("\nPredictions:")
    print("-" * 60)
    correct = 0
    for i in range(num_samples):
        is_correct = "[OK]" if predicted_digits[i] == true_labels[i] else "[X]"
        if predicted_digits[i] == true_labels[i]:
            correct += 1
        print(f"Sample {i+1}: True={true_labels[i]}, Predicted={predicted_digits[i]}, "
              f"Confidence={confidences[i]:.2%} {is_correct}")
    
    print("-" * 60)
    print(f"Accuracy: {correct}/{num_samples} ({correct/num_samples:.1%})")
    
    # Visualize predictions
    visualize_predictions(samples, true_labels, predicted_digits, confidences, 
                         save_path='new_predictions.png')
    
    return samples, true_labels, predicted_digits, confidences


def visualize_predictions(images, true_labels, predicted_labels, confidences, 
                         save_path='predictions.png', num_display=None):
    """Visualize predictions with images"""
    if num_display is None:
        num_display = len(images)
    else:
        num_display = min(num_display, len(images))
    
    fig, axes = plt.subplots(2, (num_display + 1) // 2, figsize=(15, 6))
    if num_display == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(num_display):
        ax = axes[i]
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        
        # Color code: green for correct, red for incorrect
        color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
        title = f'True: {true_labels[i]}\nPred: {predicted_labels[i]}\nConf: {confidences[i]:.1%}'
        ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_display, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPredictions visualization saved to '{save_path}'")
    plt.close()


def predict_from_files(model, file_paths):
    """Make predictions on images from files"""
    print(f"\nMaking predictions on {len(file_paths)} image(s)...")
    
    images = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        try:
            image = load_image_from_file(file_path)
            images.append(image)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not images:
        raise ValueError("No valid images to predict on")
    
    images = np.array(images)
    predicted_digits, confidences, all_predictions = predict_batch(model, images)
    
    # Display results
    print("\nPredictions:")
    print("-" * 60)
    for i, file_path in enumerate(file_paths):
        if i < len(predicted_digits):
            print(f"{os.path.basename(file_path)}: "
                  f"Predicted={predicted_digits[i]}, Confidence={confidences[i]:.2%}")
            print(f"  Probabilities: {dict(enumerate(all_predictions[i]))}")
    
    # Visualize if we have images
    if len(images) > 0:
        visualize_predictions(images, [None] * len(images), predicted_digits, confidences,
                             save_path='file_predictions.png')
    
    return predicted_digits, confidences, all_predictions


def main():
    """Main function for making predictions"""
    parser = argparse.ArgumentParser(description='Make predictions on MNIST images')
    parser.add_argument('--model', type=str, default='mnist_model.h5',
                       help='Path to saved model file')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of random test samples to predict (default: 10)')
    parser.add_argument('--files', nargs='+', type=str,
                       help='Path(s) to image file(s) to predict on')
    parser.add_argument('--test-set', action='store_true',
                       help='Use random samples from test set (default behavior)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MNIST Neural Network - Prediction Tool")
    print("=" * 60)
    
    # Load model
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("\nPlease train the model first by running: python src/main.py")
        sys.exit(1)
    
    # Make predictions
    if args.files:
        # Predict on provided image files
        predict_from_files(model, args.files)
    else:
        # Predict on random test samples
        predict_from_test_set(model, num_samples=args.samples)
    
    print("\n" + "=" * 60)
    print("Predictions completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

