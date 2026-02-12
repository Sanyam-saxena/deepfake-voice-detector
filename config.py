"""
Configuration file for Deepfake Voice Detection System
Modify these parameters to customize the system behavior.
"""

import os


class Config:
    """Central configuration class for the entire project."""
    
    # ==================== PATHS ====================
    # Base directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
    
    # Data subdirectories
    REAL_DATA_DIR = os.path.join(DATA_DIR, 'real')
    FAKE_DATA_DIR = os.path.join(DATA_DIR, 'fake')
    
    # Model paths
    MODEL_PATH = os.path.join(MODELS_DIR, 'deepfake_detector.h5')
    
    # ==================== AUDIO PROCESSING ====================
    # Sample rate (Hz)
    SAMPLE_RATE = 16000
    
    # Fixed audio duration (seconds)
    AUDIO_DURATION = 3.0
    
    # Silence trimming threshold (dB)
    SILENCE_THRESHOLD = 20
    
    # Enable/disable silence trimming
    TRIM_SILENCE = True
    
    # ==================== FEATURE EXTRACTION ====================
    # Feature type: 'mel', 'mfcc', or 'combined'
    FEATURE_TYPE = 'mel'
    
    # Mel Spectrogram parameters
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # MFCC parameters
    N_MFCC = 40
    
    # ==================== MODEL ARCHITECTURE ====================
    # Model type: 'standard' or 'advanced'
    MODEL_TYPE = 'standard'
    
    # Number of output classes (1 for binary classification)
    NUM_CLASSES = 1
    
    # Dropout rates
    DROPOUT_CONV = 0.25
    DROPOUT_DENSE = 0.5
    
    # ==================== TRAINING ====================
    # Train-test split ratio
    TEST_SIZE = 0.2
    
    # Validation split ratio (from training data)
    VALIDATION_SPLIT = 0.2
    
    # Random seed for reproducibility
    RANDOM_STATE = 42
    
    # Batch size
    BATCH_SIZE = 32
    
    # Number of epochs
    EPOCHS = 50
    
    # Learning rate
    LEARNING_RATE = 0.001
    
    # Early stopping patience
    EARLY_STOPPING_PATIENCE = 10
    
    # Learning rate reduction patience
    LR_REDUCTION_PATIENCE = 5
    
    # Learning rate reduction factor
    LR_REDUCTION_FACTOR = 0.5
    
    # Minimum learning rate
    MIN_LEARNING_RATE = 1e-7
    
    # ==================== INFERENCE ====================
    # Default confidence threshold
    CONFIDENCE_THRESHOLD = 0.7
    
    # Prediction threshold (for binary classification)
    PREDICTION_THRESHOLD = 0.5
    
    # ==================== VISUALIZATION ====================
    # Figure DPI for saved plots
    PLOT_DPI = 300
    
    # Figure size for training history
    HISTORY_FIGSIZE = (15, 10)
    
    # Figure size for confusion matrix
    CONFUSION_MATRIX_FIGSIZE = (8, 6)
    
    # Figure size for ROC curve
    ROC_CURVE_FIGSIZE = (8, 6)
    
    # ==================== STREAMLIT APP ====================
    # Page title
    PAGE_TITLE = "Deepfake Voice Detector"
    
    # Page icon
    PAGE_ICON = "🎙️"
    
    # Layout
    LAYOUT = "wide"
    
    # ==================== LOGGING ====================
    # Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    LOG_LEVEL = 'INFO'
    
    # Enable verbose output
    VERBOSE = True
    
    # ==================== SUPPORTED FORMATS ====================
    AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.REAL_DATA_DIR,
            cls.FAKE_DATA_DIR,
            cls.MODELS_DIR,
            cls.PLOTS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*80)
        print("CURRENT CONFIGURATION")
        print("="*80)
        
        sections = {
            "Audio Processing": [
                f"Sample Rate: {cls.SAMPLE_RATE} Hz",
                f"Audio Duration: {cls.AUDIO_DURATION} seconds",
                f"Trim Silence: {cls.TRIM_SILENCE}",
            ],
            "Feature Extraction": [
                f"Feature Type: {cls.FEATURE_TYPE}",
                f"N_MELS: {cls.N_MELS}",
                f"N_MFCC: {cls.N_MFCC}",
            ],
            "Model": [
                f"Model Type: {cls.MODEL_TYPE}",
                f"Num Classes: {cls.NUM_CLASSES}",
            ],
            "Training": [
                f"Batch Size: {cls.BATCH_SIZE}",
                f"Epochs: {cls.EPOCHS}",
                f"Learning Rate: {cls.LEARNING_RATE}",
                f"Test Size: {cls.TEST_SIZE}",
                f"Validation Split: {cls.VALIDATION_SPLIT}",
            ],
            "Inference": [
                f"Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}",
                f"Prediction Threshold: {cls.PREDICTION_THRESHOLD}",
            ],
        }
        
        for section, items in sections.items():
            print(f"\n{section}:")
            for item in items:
                print(f"  • {item}")
        
        print("\n" + "="*80 + "\n")
    
    @classmethod
    def validate_config(cls):
        """Validate configuration parameters."""
        errors = []
        
        # Check sample rate
        if cls.SAMPLE_RATE <= 0:
            errors.append("SAMPLE_RATE must be positive")
        
        # Check audio duration
        if cls.AUDIO_DURATION <= 0:
            errors.append("AUDIO_DURATION must be positive")
        
        # Check feature type
        if cls.FEATURE_TYPE not in ['mel', 'mfcc', 'combined']:
            errors.append("FEATURE_TYPE must be 'mel', 'mfcc', or 'combined'")
        
        # Check batch size
        if cls.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive")
        
        # Check epochs
        if cls.EPOCHS <= 0:
            errors.append("EPOCHS must be positive")
        
        # Check learning rate
        if cls.LEARNING_RATE <= 0:
            errors.append("LEARNING_RATE must be positive")
        
        # Check test size
        if not 0 < cls.TEST_SIZE < 1:
            errors.append("TEST_SIZE must be between 0 and 1")
        
        # Check validation split
        if not 0 < cls.VALIDATION_SPLIT < 1:
            errors.append("VALIDATION_SPLIT must be between 0 and 1")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  • {e}" for e in errors))
        
        return True


# Create an instance for easy import
config = Config()


if __name__ == "__main__":
    # When run directly, print and validate configuration
    config.validate_config()
    config.print_config()
