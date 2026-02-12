"""
Deepfake Voice Detection System
Source code package containing preprocessing, feature extraction, and model modules.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easy access
from .preprocess import (
    load_audio,
    normalize_audio,
    trim_silence,
    preprocess_audio,
    load_dataset_from_directory
)

from .features import (
    extract_mel_spectrogram,
    extract_mfcc,
    extract_combined_features,
    prepare_cnn_input,
    extract_features_from_audio_list
)

from .model import (
    create_cnn_model,
    create_advanced_cnn_model,
    compile_model,
    save_model,
    load_saved_model
)

__all__ = [
    # Preprocessing
    'load_audio',
    'normalize_audio',
    'trim_silence',
    'preprocess_audio',
    'load_dataset_from_directory',
    
    # Features
    'extract_mel_spectrogram',
    'extract_mfcc',
    'extract_combined_features',
    'prepare_cnn_input',
    'extract_features_from_audio_list',
    
    # Model
    'create_cnn_model',
    'create_advanced_cnn_model',
    'compile_model',
    'save_model',
    'load_saved_model',
]
