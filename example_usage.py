"""
Example Usage Script
Demonstrates how to use the deepfake detection system programmatically.
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import preprocess_audio
from src.features import extract_mel_spectrogram, prepare_cnn_input, pad_features, normalize_features
from src.model import load_saved_model


def predict_single_audio(audio_path, model_path='models/deepfake_detector.h5'):
    """
    Predict whether an audio file is real or fake.
    
    Args:
        audio_path (str): Path to audio file
        model_path (str): Path to trained model
    
    Returns:
        dict: Prediction results
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {audio_path}")
    print(f"{'='*80}\n")
    
    # Step 1: Load model
    print("Loading model...")
    model = load_saved_model(model_path)
    
    # Get target width from model input shape
    target_width = model.input_shape[2]
    
    # Step 2: Preprocess audio
    print("Preprocessing audio...")
    audio = preprocess_audio(audio_path, sr=16000, duration=3.0)
    print(f"Audio shape: {audio.shape}")
    
    # Step 3: Extract features
    print("Extracting features...")
    spectrogram = extract_mel_spectrogram(audio, sr=16000)
    print(f"Spectrogram shape: {spectrogram.shape}")
    
    # Step 4: Prepare for CNN
    features = prepare_cnn_input(spectrogram)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    
    # Step 5: Pad features
    features = pad_features(features, target_width)
    
    # Step 6: Normalize
    features = normalize_features(features)
    print(f"Final feature shape: {features.shape}")
    
    # Step 7: Make prediction
    print("\nMaking prediction...")
    prediction_proba = model.predict(features, verbose=0)[0][0]
    prediction = 1 if prediction_proba > 0.5 else 0
    confidence = prediction_proba if prediction == 1 else (1 - prediction_proba)
    
    # Step 8: Display results
    print(f"\n{'='*80}")
    print("PREDICTION RESULTS")
    print(f"{'='*80}")
    
    result = {
        'file': os.path.basename(audio_path),
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'confidence': confidence * 100,
        'raw_score': prediction_proba
    }
    
    print(f"File: {result['file']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Raw Score: {result['raw_score']:.4f}")
    
    if confidence >= 0.7:
        print("Status: ✅ High confidence")
    else:
        print("Status: ⚠️  Low confidence")
    
    print(f"{'='*80}\n")
    
    return result


def batch_predict(audio_dir, model_path='models/deepfake_detector.h5'):
    """
    Predict multiple audio files in a directory.
    
    Args:
        audio_dir (str): Directory containing audio files
        model_path (str): Path to trained model
    
    Returns:
        list: List of prediction results
    """
    print(f"\n{'='*80}")
    print(f"BATCH PREDICTION")
    print(f"{'='*80}\n")
    
    # Get all audio files
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    audio_files = [f for f in os.listdir(audio_dir) 
                   if f.lower().endswith(audio_extensions)]
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return []
    
    print(f"Found {len(audio_files)} audio files\n")
    
    results = []
    for i, audio_file in enumerate(audio_files, 1):
        audio_path = os.path.join(audio_dir, audio_file)
        print(f"[{i}/{len(audio_files)}] Processing {audio_file}...")
        
        try:
            result = predict_single_audio(audio_path, model_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total files: {len(results)}")
    print(f"Real: {sum(1 for r in results if r['prediction'] == 'REAL')}")
    print(f"Fake: {sum(1 for r in results if r['prediction'] == 'FAKE')}")
    print(f"{'='*80}\n")
    
    return results


def main():
    """Main function demonstrating usage."""
    print("\n" + "="*80)
    print("DEEPFAKE VOICE DETECTION - EXAMPLE USAGE")
    print("="*80 + "\n")
    
    # Check if model exists
    model_path = 'models/deepfake_detector.h5'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using: python train.py")
        return
    
    print("Choose an option:")
    print("1. Predict single audio file")
    print("2. Batch predict directory")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        # Single file prediction
        audio_path = input("Enter path to audio file: ").strip()
        if os.path.exists(audio_path):
            predict_single_audio(audio_path, model_path)
        else:
            print(f"❌ File not found: {audio_path}")
    
    elif choice == '2':
        # Batch prediction
        audio_dir = input("Enter directory path: ").strip()
        if os.path.exists(audio_dir):
            batch_predict(audio_dir, model_path)
        else:
            print(f"❌ Directory not found: {audio_dir}")
    
    elif choice == '3':
        print("Exiting...")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
