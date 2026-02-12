"""
Audio Preprocessing Module
Handles loading, resampling, normalization, and trimming of audio files.
"""

import librosa
import numpy as np
import os
from typing import Tuple, Optional


def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.
    
    Args:
        file_path (str): Path to the audio file
        sr (int): Target sample rate (default: 16000 Hz)
    
    Returns:
        Tuple[np.ndarray, int]: Audio time series and sample rate
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: If audio file cannot be loaded
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        raise Exception(f"Error loading audio file {file_path}: {str(e)}")


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to have values between -1 and 1.
    
    Args:
        audio (np.ndarray): Input audio signal
    
    Returns:
        np.ndarray: Normalized audio signal
    """
    if len(audio) == 0:
        return audio
    
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    return audio


def trim_silence(audio: np.ndarray, sr: int = 16000, 
                 top_db: int = 20) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.
    
    Args:
        audio (np.ndarray): Input audio signal
        sr (int): Sample rate
        top_db (int): Threshold (in dB) below reference to consider as silence
    
    Returns:
        np.ndarray: Trimmed audio signal
    """
    try:
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio_trimmed
    except Exception as e:
        print(f"Warning: Could not trim silence: {str(e)}")
        return audio


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate audio to a fixed length.
    
    Args:
        audio (np.ndarray): Input audio signal
        target_length (int): Target length in samples
    
    Returns:
        np.ndarray: Audio signal with fixed length
    """
    current_length = len(audio)
    
    if current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode='constant')
    elif current_length > target_length:
        # Truncate
        audio = audio[:target_length]
    
    return audio


def preprocess_audio(file_path: str, sr: int = 16000, 
                     duration: Optional[float] = None,
                     trim_silence_flag: bool = True) -> np.ndarray:
    """
    Complete preprocessing pipeline for audio files.
    
    Args:
        file_path (str): Path to audio file
        sr (int): Target sample rate
        duration (float, optional): Fixed duration in seconds
        trim_silence_flag (bool): Whether to trim silence
    
    Returns:
        np.ndarray: Preprocessed audio signal
    """
    # Load audio
    audio, _ = load_audio(file_path, sr=sr)
    
    # Trim silence if requested
    if trim_silence_flag:
        audio = trim_silence(audio, sr=sr)
    
    # Normalize
    audio = normalize_audio(audio)
    
    # Pad or truncate to fixed duration if specified
    if duration is not None:
        target_length = int(sr * duration)
        audio = pad_or_truncate(audio, target_length)
    
    return audio


def load_dataset_from_directory(data_dir: str, sr: int = 16000,
                                duration: Optional[float] = None) -> Tuple[list, list]:
    """
    Load all audio files from a directory structure (data/real/ and data/fake/).
    
    Args:
        data_dir (str): Base directory containing 'real' and 'fake' subdirectories
        sr (int): Target sample rate
        duration (float, optional): Fixed duration in seconds
    
    Returns:
        Tuple[list, list]: List of audio arrays and corresponding labels (0=real, 1=fake)
    """
    audio_data = []
    labels = []
    
    # Define class directories
    classes = {
        'real': 0,
        'fake': 1
    }
    
    for class_name, label in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping...")
            continue
        
        # Get all audio files
        audio_files = [f for f in os.listdir(class_dir) 
                      if f.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
        
        print(f"Loading {len(audio_files)} files from {class_name}...")
        
        for audio_file in audio_files:
            file_path = os.path.join(class_dir, audio_file)
            try:
                audio = preprocess_audio(file_path, sr=sr, duration=duration)
                audio_data.append(audio)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
    
    return audio_data, labels
