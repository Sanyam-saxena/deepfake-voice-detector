"""
Feature Extraction Module
Extracts audio features like Mel Spectrograms and MFCCs.
"""

import librosa
import librosa.display
import numpy as np
from typing import Tuple


def extract_mel_spectrogram(audio: np.ndarray, sr: int = 16000,
                           n_mels: int = 128, n_fft: int = 2048,
                           hop_length: int = 512) -> np.ndarray:
    """
    Extract Mel Spectrogram from audio signal.
    
    Args:
        audio (np.ndarray): Input audio signal
        sr (int): Sample rate
        n_mels (int): Number of mel bands
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
    
    Returns:
        np.ndarray: Mel spectrogram (in dB scale)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def extract_mfcc(audio: np.ndarray, sr: int = 16000,
                n_mfcc: int = 40, n_fft: int = 2048,
                hop_length: int = 512) -> np.ndarray:
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs) from audio.
    
    Args:
        audio (np.ndarray): Input audio signal
        sr (int): Sample rate
        n_mfcc (int): Number of MFCCs to extract
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
    
    Returns:
        np.ndarray: MFCC features
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return mfcc


def extract_combined_features(audio: np.ndarray, sr: int = 16000,
                             n_mels: int = 128, n_mfcc: int = 40) -> np.ndarray:
    """
    Extract both Mel Spectrogram and MFCCs and combine them.
    
    Args:
        audio (np.ndarray): Input audio signal
        sr (int): Sample rate
        n_mels (int): Number of mel bands
        n_mfcc (int): Number of MFCCs
    
    Returns:
        np.ndarray: Combined feature matrix
    """
    # Extract mel spectrogram
    mel_spec = extract_mel_spectrogram(audio, sr=sr, n_mels=n_mels)
    
    # Extract MFCCs
    mfcc = extract_mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    
    # Combine features
    combined = np.vstack([mel_spec, mfcc])
    
    return combined


def prepare_cnn_input(features: np.ndarray) -> np.ndarray:
    """
    Prepare feature matrix for CNN input by adding channel dimension.
    
    Args:
        features (np.ndarray): Feature matrix (height x width)
    
    Returns:
        np.ndarray: Feature matrix with channel dimension (height x width x 1)
    """
    # Add channel dimension for CNN
    return np.expand_dims(features, axis=-1)


def extract_features_from_audio_list(audio_list: list, sr: int = 16000,
                                     feature_type: str = 'mel') -> np.ndarray:
    """
    Extract features from a list of audio signals.
    
    Args:
        audio_list (list): List of audio signals
        sr (int): Sample rate
        feature_type (str): Type of features ('mel', 'mfcc', or 'combined')
    
    Returns:
        np.ndarray: Array of feature matrices
    """
    features = []
    
    for audio in audio_list:
        if feature_type == 'mel':
            feat = extract_mel_spectrogram(audio, sr=sr)
        elif feature_type == 'mfcc':
            feat = extract_mfcc(audio, sr=sr)
        elif feature_type == 'combined':
            feat = extract_combined_features(audio, sr=sr)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Prepare for CNN
        feat = prepare_cnn_input(feat)
        features.append(feat)
    
    return np.array(features)


def pad_features(features: np.ndarray, target_width: int) -> np.ndarray:
    """
    Pad or truncate features to a target width (time dimension).
    
    Args:
        features (np.ndarray): Feature array (batch x height x width x channels)
        target_width (int): Target width
    
    Returns:
        np.ndarray: Padded/truncated features
    """
    batch_size, height, current_width, channels = features.shape
    
    if current_width < target_width:
        # Pad
        padding = target_width - current_width
        features = np.pad(features, 
                         ((0, 0), (0, 0), (0, padding), (0, 0)),
                         mode='constant',
                         constant_values=0)
    elif current_width > target_width:
        # Truncate
        features = features[:, :, :target_width, :]
    
    return features


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to [0, 1] range.
    
    Args:
        features (np.ndarray): Input features
    
    Returns:
        np.ndarray: Normalized features
    """
    # Normalize each sample independently
    normalized = np.zeros_like(features)
    
    for i in range(features.shape[0]):
        sample = features[i]
        min_val = np.min(sample)
        max_val = np.max(sample)
        
        if max_val > min_val:
            normalized[i] = (sample - min_val) / (max_val - min_val)
        else:
            normalized[i] = sample
    
    return normalized
