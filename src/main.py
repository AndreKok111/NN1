"""
Simple Neural Network for MNIST using TensorFlow
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


def load_and_preprocess_data():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data for CNN (28, 28, 1) or flatten for Dense layers
    # Using flatten for a simple neural network
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    
    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image shape: {x_train.shape[1]}")
    
    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Build a simple neural network model"""
    print("\nBuilding neural network model...")
    
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    return model


def train_model(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=128):
    """Train the neural network"""
    print(f"\nTraining model for {epochs} epochs...")
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    return history


def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model"""
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    return test_loss, test_accuracy


def plot_training_history(history):
    """Plot training history"""
    print("\nPlotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to 'training_history.png'")
    plt.close()


def predict_sample(model, x_test, y_test, num_samples=5):
    """Make predictions on sample test images"""
    print(f"\nMaking predictions on {num_samples} sample images...")
    
    # Get random samples
    indices = np.random.choice(x_test.shape[0], num_samples, replace=False)
    samples = x_test[indices]
    true_labels = np.argmax(y_test[indices], axis=1)
    
    # Make predictions
    predictions = model.predict(samples, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Display results
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'True: {true_labels[i]}\nPred: {predicted_labels[i]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    print("Sample predictions saved to 'sample_predictions.png'")
    plt.close()


def main():
    """Main function to run the neural network"""
    print("=" * 60)
    print("MNIST Neural Network with TensorFlow")
    print("=" * 60)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=5)
    
    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Make sample predictions
    predict_sample(model, x_test, y_test, num_samples=5)
    
    # Save model
    model.save('mnist_model.h5')
    print("\nModel saved to 'mnist_model.h5'")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

