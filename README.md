# NN1 - MNIST Neural Network

A simple neural network built with TensorFlow to classify handwritten digits from the MNIST dataset.

## Features

- Simple feedforward neural network with dropout layers
- Trains on MNIST dataset (60,000 training images, 10,000 test images)
- Visualizes training history and sample predictions
- Saves trained model for later use
- **Make predictions on new images** - Load saved model and predict on custom images or test samples

## Model Architecture

- Input Layer: 784 neurons (28x28 flattened images)
- Hidden Layer 1: 128 neurons with ReLU activation + Dropout (0.2)
- Hidden Layer 2: 64 neurons with ReLU activation + Dropout (0.2)
- Output Layer: 10 neurons with Softmax activation (one for each digit 0-9)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda install tensorflow numpy matplotlib
```

## Usage

### Training the Model

Run the training script:
```bash
python src/main.py
```

The script will:
1. Load and preprocess the MNIST dataset
2. Build and compile the neural network
3. Train the model for 5 epochs
4. Evaluate on the test set
5. Generate visualization plots:
   - `training_history.png` - Training/validation accuracy and loss curves
   - `sample_predictions.png` - Sample predictions on test images
6. Save the trained model as `mnist_model.h5`

### Making Predictions on New Samples

After training, you can make predictions on new images using the saved model:

**Predict on random test samples:**
```bash
python src/predict.py --samples 10
```

**Predict on custom image files:**
```bash
python src/predict.py --files path/to/image1.png path/to/image2.jpg
```

**Predict on images from NMIST directory:**
```bash
python src/predict_nmist.py
```

This script automatically loads all images from the NMIST directory (`C:\Users\andre\OneDrive\Documenten\Andre\PHK Academie\NMIST`) and makes predictions on them. It recognizes Dutch number names in filenames (een=1, twee=2, etc.) and compares predictions with expected values.

**Options for `predict.py`:**
- `--model PATH` - Path to saved model file (default: `mnist_model.h5`)
- `--samples N` - Number of random test samples to predict (default: 10)
- `--files IMAGE1 IMAGE2 ...` - Path(s) to image file(s) to predict on
- `--test-set` - Use random samples from test set (default behavior)

**Options for `predict_nmist.py`:**
- `--model PATH` - Path to saved model file (default: `mnist_model.h5`)
- `--dir PATH` - Path to NMIST directory (default: `C:\Users\andre\OneDrive\Documenten\Andre\PHK Academie\NMIST`)

The prediction scripts will:
- Load the saved model
- Make predictions on the provided images
- Display predictions with confidence scores
- Generate visualization showing predictions (`new_predictions.png`, `file_predictions.png`, or `nmist_predictions.png`)

**Note:** Custom images should be:
- Grayscale or RGB (will be converted automatically)
- Any size (will be resized to 28x28)
- Preferably with white background and black digits (MNIST format)

## Expected Results

After training, you should see:
- Test accuracy around 95-98%
- Training and validation plots
- Sample predictions showing correct classifications

## Requirements

- Python 3.9+
- TensorFlow 2.13+
- NumPy 1.24+
- Matplotlib 3.7+
- Pillow 9.0+ (for image processing in predictions)

