"""
Training Script for Deepfake Voice Detection
Handles data loading, preprocessing, model training, and evaluation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
import keras

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import load_dataset_from_directory
from src.features import extract_features_from_audio_list, pad_features, normalize_features
from src.model import create_cnn_model, compile_model, get_model_summary, save_model


# Configuration
CONFIG = {
    'data_dir': 'data',
    'sample_rate': 16000,
    'audio_duration': 3.0,  # seconds
    'feature_type': 'mel',  # 'mel', 'mfcc', or 'combined'
    'test_size': 0.2,
    'validation_split': 0.2,
    'random_state': 42,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'model_path': 'models/deepfake_detector.h5',
    'plots_dir': 'plots'
}


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('models', exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)


def plot_training_history(history, save_path='plots/training_history.png'):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Keras training history object
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='plots/confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path='plots/roc_curve.png'):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path (str): Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and print detailed metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Get predictions
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Real', 'Fake'],
                               digits=4))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba)
    
    # Calculate and print additional metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    print("\nSummary Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("="*80 + "\n")
    
    return metrics


def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("DEEPFAKE VOICE DETECTION - TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Create directories
    create_directories()
    
    # Set random seeds for reproducibility
    np.random.seed(CONFIG['random_state'])
    tf.random.set_seed(CONFIG['random_state'])
    
    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    audio_data, labels = load_dataset_from_directory(
        CONFIG['data_dir'],
        sr=CONFIG['sample_rate'],
        duration=CONFIG['audio_duration']
    )
    
    print(f"Total samples loaded: {len(audio_data)}")
    print(f"Real samples: {labels.count(0)}")
    print(f"Fake samples: {labels.count(1)}")
    
    if len(audio_data) == 0:
        print("\nERROR: No audio files found!")
        print(f"Please ensure audio files are placed in:")
        print(f"  - {CONFIG['data_dir']}/real/")
        print(f"  - {CONFIG['data_dir']}/fake/")
        return
    
    # Step 2: Extract features
    print("\nStep 2: Extracting features...")
    features = extract_features_from_audio_list(
        audio_data,
        sr=CONFIG['sample_rate'],
        feature_type=CONFIG['feature_type']
    )
    
    print(f"Feature shape before padding: {features.shape}")
    
    # Find maximum width and pad all features
    max_width = max([f.shape[1] for f in features])
    features = pad_features(features, max_width)
    
    # Normalize features
    features = normalize_features(features)
    
    print(f"Feature shape after padding: {features.shape}")
    
    # Convert labels to numpy array
    labels = np.array(labels)
    
    # Step 3: Split dataset
    print("\nStep 3: Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Step 4: Create model
    print("\nStep 4: Creating model...")
    input_shape = X_train.shape[1:]  # (height, width, channels)
    model = create_cnn_model(input_shape)
    model = compile_model(model, learning_rate=CONFIG['learning_rate'])
    get_model_summary(model)
    
    # Step 5: Train model
    print("Step 5: Training model...")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            CONFIG['model_path'],
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        validation_split=CONFIG['validation_split'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 6: Plot training history
    print("\nStep 6: Plotting training history...")
    plot_training_history(history)
    
    # Step 7: Evaluate model
    print("\nStep 7: Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 8: Save final model
    print("\nStep 8: Saving model...")
    save_model(model, CONFIG['model_path'])
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {CONFIG['model_path']}")
    print(f"Plots saved to: {CONFIG['plots_dir']}/")
    print("\nYou can now run the Streamlit app using:")
    print("  streamlit run app.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
