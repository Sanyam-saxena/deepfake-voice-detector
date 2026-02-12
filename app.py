"""
Streamlit Web Application for Deepfake Voice Detection
Allows users to upload audio files and get real-time predictions.
"""

import os
import sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import keras

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import preprocess_audio
from src.features import extract_mel_spectrogram, prepare_cnn_input, pad_features, normalize_features


# Page configuration
st.set_page_config(
    page_title="Deepfake Voice Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .real-voice {
        background-color: #D4EDDA;
        color: #155724;
        border: 2px solid #C3E6CB;
    }
    .fake-voice {
        background-color: #F8D7DA;
        color: #721C24;
        border: 2px solid #F5C6CB;
    }
    .confidence-score {
        font-size: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path='models/deepfake_detector.h5'):
    """Load the trained model."""
    try:
        model = keras.saving.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def plot_waveform(audio, sr, title="Audio Waveform"):
    """Plot audio waveform."""
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spectrogram(spectrogram, sr=16000, title="Mel Spectrogram"):
    """Plot mel spectrogram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(
        spectrogram,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap='viridis'
    )
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Mel Frequency", fontsize=12)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


def predict_audio(model, audio_file, target_width):
    """
    Process audio and make prediction using chunk-based analysis.
    
    Splits longer audio (up to 60s) into 3-second chunks, predicts each
    chunk independently, and aggregates results.
    
    Args:
        model: Trained Keras model
        audio_file: Uploaded audio file
        target_width: Target width for padding
    
    Returns:
        tuple: (prediction, confidence, audio, sr, spectrogram, chunk_results)
    """
    import librosa as lr
    
    # Save uploaded file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    sr = 16000
    chunk_duration = 3.0  # seconds per chunk (model's expected input)
    max_duration = 60.0   # max 1 minute
    chunk_samples = int(sr * chunk_duration)
    
    # Load full audio (up to 60 seconds)
    full_audio, _ = lr.load(temp_path, sr=sr, duration=max_duration)
    
    # Trim silence from the full audio
    full_audio, _ = lr.effects.trim(full_audio, top_db=20)
    
    # If audio is shorter than one chunk, pad it
    if len(full_audio) < chunk_samples:
        full_audio = np.pad(full_audio, (0, chunk_samples - len(full_audio)))
    
    # Split into 3-second chunks
    num_chunks = len(full_audio) // chunk_samples
    chunks = [full_audio[i * chunk_samples:(i + 1) * chunk_samples] for i in range(num_chunks)]
    
    # Predict each chunk
    chunk_results = []
    all_spectrograms = []
    
    for i, chunk in enumerate(chunks):
        spectrogram = extract_mel_spectrogram(chunk, sr=sr)
        all_spectrograms.append(spectrogram)
        features = prepare_cnn_input(spectrogram)
        features = np.expand_dims(features, axis=0)
        features = pad_features(features, target_width)
        features = normalize_features(features)
        
        proba = model.predict(features, verbose=0)[0][0]
        chunk_pred = 1 if proba > 0.5 else 0
        chunk_conf = proba if chunk_pred == 1 else (1 - proba)
        
        chunk_results.append({
            'chunk': i + 1,
            'start': i * chunk_duration,
            'end': (i + 1) * chunk_duration,
            'prediction': 'Fake' if chunk_pred == 1 else 'Real',
            'confidence': chunk_conf,
            'raw_score': proba
        })
    
    # Aggregate: average raw scores across all chunks
    avg_score = np.mean([r['raw_score'] for r in chunk_results])
    prediction = 1 if avg_score > 0.5 else 0
    confidence = avg_score if prediction == 1 else (1 - avg_score)
    
    # Use the first chunk's spectrogram for display
    spectrogram = all_spectrograms[0]
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return prediction, confidence, full_audio, sr, spectrogram, chunk_results


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎙️ Deepfake Voice Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered system to detect deepfake and synthetic voices</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.write("""
        This application uses a Convolutional Neural Network (CNN) to classify audio as:
        - **Real**: Authentic human voice
        - **Fake**: AI-generated/Deepfake voice
        """)
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence required for a definitive prediction"
        )
        
        st.header("ℹ️ Instructions")
        st.write("""
        1. Upload an audio file (WAV, MP3, etc.)
        2. Audio up to **60 seconds** is supported
        3. Audio is split into 3-second chunks for analysis
        4. View per-chunk and overall predictions
        """)
        
        st.header("🔧 Technical Details")
        st.write("""
        - **Model**: CNN with Batch Normalization
        - **Features**: Mel Spectrogram
        - **Sample Rate**: 16 kHz
        - **Max Duration**: 60 seconds
        - **Analysis**: Chunk-based (3s segments)
        """)
    
    # Main content
    st.header("📤 Upload Audio File")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        help="Supported formats: WAV, MP3, FLAC, OGG, M4A"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Load model
        with st.spinner("Loading model..."):
            model = load_model()
        
        if model is None:
            st.error("❌ Model not found! Please train the model first using `python train.py`")
            st.stop()
        
        # Get target width from model input shape
        target_width = model.input_shape[2]
        
        # Make prediction
        with st.spinner("🔍 Analyzing audio..."):
            try:
                prediction, confidence, audio, sr, spectrogram, chunk_results = predict_audio(
                    model, uploaded_file, target_width
                )
                
                # Display results
                st.header("📈 Analysis Results")
                
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎵 Waveform")
                    fig_wave = plot_waveform(audio, sr)
                    st.pyplot(fig_wave)
                    plt.close()
                
                with col2:
                    st.subheader("🔊 Mel Spectrogram")
                    fig_spec = plot_spectrogram(spectrogram, sr)
                    st.pyplot(fig_spec)
                    plt.close()
                
                # Prediction result
                st.header("🎯 Prediction Result")
                
                # Determine prediction label and style
                if prediction == 0:
                    label = "✅ REAL VOICE"
                    css_class = "real-voice"
                    emoji = "👤"
                else:
                    label = "⚠️ DEEPFAKE DETECTED"
                    css_class = "fake-voice"
                    emoji = "🤖"
                
                # Display prediction with styling
                st.markdown(
                    f'<div class="prediction-box {css_class}">'
                    f'{emoji} {label}'
                    f'<div class="confidence-score">Confidence: {confidence*100:.2f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Confidence interpretation
                if confidence >= confidence_threshold:
                    st.success(f"✅ High confidence prediction (≥ {confidence_threshold*100:.0f}%)")
                else:
                    st.warning(f"⚠️ Low confidence prediction (< {confidence_threshold*100:.0f}%). "
                             "Results may be uncertain.")
                
                # Additional metrics
                st.header("📊 Detailed Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", "Real" if prediction == 0 else "Fake")
                
                with col2:
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                
                with col3:
                    raw_score = confidence if prediction == 1 else (1 - confidence)
                    st.metric("Raw Score", f"{raw_score:.4f}")
                
                # Per-chunk analysis
                if len(chunk_results) > 1:
                    st.header("🧩 Per-Chunk Analysis")
                    st.write(f"Audio split into **{len(chunk_results)} chunks** of 3 seconds each:")
                    
                    import pandas as pd
                    chunk_df = pd.DataFrame(chunk_results)
                    chunk_df['Time Range'] = chunk_df.apply(
                        lambda r: f"{r['start']:.0f}s - {r['end']:.0f}s", axis=1
                    )
                    chunk_df['Confidence'] = chunk_df['confidence'].apply(
                        lambda c: f"{c*100:.1f}%"
                    )
                    display_df = chunk_df[['chunk', 'Time Range', 'prediction', 'Confidence']]
                    display_df.columns = ['Chunk #', 'Time Range', 'Prediction', 'Confidence']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Summary stats
                    fake_chunks = sum(1 for r in chunk_results if r['prediction'] == 'Fake')
                    real_chunks = sum(1 for r in chunk_results if r['prediction'] == 'Real')
                    st.write(f"**Real chunks**: {real_chunks} | **Fake chunks**: {fake_chunks}")
                
                # Audio info
                with st.expander("🔍 Audio Information"):
                    st.write(f"**Duration**: {len(audio)/sr:.2f} seconds")
                    st.write(f"**Sample Rate**: {sr} Hz")
                    st.write(f"**Samples**: {len(audio)}")
                    st.write(f"**Chunks Analyzed**: {len(chunk_results)}")
                    st.write(f"**Spectrogram Shape**: {spectrogram.shape}")
                
                # Play audio
                st.header("🔊 Play Audio")
                st.audio(uploaded_file, format='audio/wav')
                
            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")
                st.exception(e)
    
    else:
        # Show placeholder when no file is uploaded
        st.info("👆 Please upload an audio file to begin analysis")
        
        # Example usage
        with st.expander("📖 Example Usage"):
            st.write("""
            **Step 1**: Prepare your audio file
            - Ensure it's in a supported format (WAV, MP3, etc.)
            - Any duration is acceptable (will be processed to 3 seconds)
            
            **Step 2**: Upload the file
            - Click "Browse files" above
            - Select your audio file
            
            **Step 3**: View results
            - Waveform visualization
            - Mel Spectrogram
            - Prediction: Real or Fake
            - Confidence score
            
            **Step 4**: Interpret results
            - Green box = Real voice detected
            - Red box = Deepfake detected
            - Check confidence score for reliability
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit, TensorFlow, and Librosa | "
        "Deepfake Voice Detection System"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
