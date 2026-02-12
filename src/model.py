"""
Model Architecture Module
Defines CNN-based models for deepfake voice detection.
"""

import keras
from keras import layers, models
from typing import Tuple


def create_cnn_model(input_shape: Tuple[int, int, int],
                     num_classes: int = 1) -> keras.Model:
    """
    Create a CNN model for audio classification.
    
    Args:
        input_shape (Tuple[int, int, int]): Input shape (height, width, channels)
        num_classes (int): Number of output classes (1 for binary classification)
    
    Returns:
        keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model


def create_advanced_cnn_model(input_shape: Tuple[int, int, int],
                              num_classes: int = 1) -> keras.Model:
    """
    Create an advanced CNN model with residual connections.
    
    Args:
        input_shape (Tuple[int, int, int]): Input shape (height, width, channels)
        num_classes (int): Number of output classes
    
    Returns:
        keras.Model: Compiled advanced CNN model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv Block
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual Block 1
    residual = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    
    # Residual Block 2
    residual = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Residual Block 3
    residual = layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense Layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model


def compile_model(model: keras.Model, 
                 learning_rate: float = 0.001) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model (keras.Model): Model to compile
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def get_model_summary(model: keras.Model) -> None:
    """
    Print model architecture summary.
    
    Args:
        model (keras.Model): Model to summarize
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    model.summary()
    print("="*80 + "\n")


def save_model(model: keras.Model, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model (keras.Model): Trained model
        filepath (str): Path to save the model
    """
    try:
        model.save(filepath)
        print(f"Model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")


def load_saved_model(filepath: str) -> keras.Model:
    """
    Load a saved model from disk.
    
    Args:
        filepath (str): Path to the saved model
    
    Returns:
        keras.Model: Loaded model
    """
    try:
        model = keras.saving.load_model(filepath)
        print(f"Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
