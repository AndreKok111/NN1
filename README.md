# NN1 - MNIST Neural Network

A simple neural network built with TensorFlow to classify handwritten digits from the MNIST dataset.

## Features

- Simple feedforward neural network with dropout layers
- Trains on MNIST dataset (60,000 training images, 10,000 test images)
- Visualizes training history and sample predictions
- Saves trained model for later use

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

