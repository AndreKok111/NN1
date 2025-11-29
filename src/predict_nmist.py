"""
Make predictions on images from the NMIST directory
"""

import os
import sys
from predict import load_model, predict_from_files, visualize_predictions
import numpy as np

# Map Dutch number names to digits
DUTCH_TO_DIGIT = {
    'een': 1,
    'twee': 2,
    'drie': 3,
    'vier': 4,
    'vijf': 5,
    'zes': 6,
    'zeven': 7,
    'acht': 8,
    'negen': 9
}


def get_nmist_images(nmist_dir):
    """Get all image files from the NMIST directory"""
    if not os.path.exists(nmist_dir):
        raise FileNotFoundError(f"NMIST directory not found: {nmist_dir}")
    
    # Get all PNG files
    image_files = []
    for file in os.listdir(nmist_dir):
        if file.lower().endswith('.png'):
            image_files.append(os.path.join(nmist_dir, file))
    
    # Sort by filename to maintain order
    image_files.sort()
    
    return image_files


def get_expected_label(filename):
    """Extract expected label from filename (Dutch number name)"""
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0].lower()
    return DUTCH_TO_DIGIT.get(name_without_ext, None)


def predict_nmist_samples(model_path='mnist_model.h5', 
                          nmist_dir=r"C:\Users\andre\OneDrive\Documenten\Andre\PHK Academie\NMIST"):
    """Make predictions on all images in the NMIST directory"""
    print("=" * 60)
    print("MNIST Neural Network - Predicting on NMIST Directory Samples")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    try:
        model = load_model(model_path)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("\nPlease train the model first by running: python src/main.py")
        sys.exit(1)
    
    # Get all image files from NMIST directory
    print(f"\nScanning NMIST directory: {nmist_dir}")
    try:
        image_files = get_nmist_images(nmist_dir)
        print(f"Found {len(image_files)} image(s)")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    
    if not image_files:
        print("No image files found in the NMIST directory!")
        sys.exit(1)
    
    # Display files found
    print("\nFiles to process:")
    for i, file_path in enumerate(image_files, 1):
        basename = os.path.basename(file_path)
        expected = get_expected_label(file_path)
        expected_str = f" (expected: {expected})" if expected is not None else ""
        print(f"  {i}. {basename}{expected_str}")
    
    # Make predictions
    print("\n" + "-" * 60)
    print("Making predictions...")
    print("-" * 60)
    
    from predict import load_image_from_file, predict_batch
    import numpy as np
    
    images = []
    file_basenames = []
    expected_labels = []
    
    for file_path in image_files:
        try:
            image = load_image_from_file(file_path)
            images.append(image)
            file_basenames.append(os.path.basename(file_path))
            expected_labels.append(get_expected_label(file_path))
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue
    
    if not images:
        print("No valid images could be loaded!")
        sys.exit(1)
    
    # Make batch predictions
    images_array = np.array(images)
    predicted_digits, confidences, all_predictions = predict_batch(model, images_array)
    
    # Display results
    print("\nPrediction Results:")
    print("=" * 60)
    correct = 0
    total = len(predicted_digits)
    
    for i in range(total):
        filename = file_basenames[i]
        predicted = predicted_digits[i]
        confidence = confidences[i]
        expected = expected_labels[i]
        
        # Check if prediction matches expected (if we have expected label)
        if expected is not None:
            is_correct = predicted == expected
            status = "[OK]" if is_correct else "[X]"
            if is_correct:
                correct += 1
            print(f"{filename:15} -> Predicted: {predicted} (Confidence: {confidence:.2%}) "
                  f"Expected: {expected} {status}")
        else:
            print(f"{filename:15} -> Predicted: {predicted} (Confidence: {confidence:.2%})")
    
    print("=" * 60)
    
    if any(e is not None for e in expected_labels):
        print(f"\nAccuracy: {correct}/{total} correct ({correct/total:.1%})")
    
    # Create visualization
    print("\nGenerating visualization...")
    # Use expected labels if available, otherwise use None
    true_labels = expected_labels if all(e is not None for e in expected_labels) else [None] * len(images)
    visualize_predictions(images_array, true_labels, predicted_digits, confidences,
                         save_path='nmist_predictions.png')
    
    print("\n" + "=" * 60)
    print("Predictions completed!")
    print("=" * 60)
    print(f"\nVisualization saved to: nmist_predictions.png")
    
    return predicted_digits, confidences, all_predictions


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions on NMIST directory images')
    parser.add_argument('--model', type=str, default='mnist_model.h5',
                       help='Path to saved model file (default: mnist_model.h5)')
    parser.add_argument('--dir', type=str, 
                       default=r"C:\Users\andre\OneDrive\Documenten\Andre\PHK Academie\NMIST",
                       help='Path to NMIST directory containing images')
    
    args = parser.parse_args()
    
    predict_nmist_samples(model_path=args.model, nmist_dir=args.dir)


if __name__ == "__main__":
    main()

