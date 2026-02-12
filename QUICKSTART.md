# 🚀 Quick Start Guide

Get up and running with the Deepfake Voice Detection System in 5 minutes!

## ⚡ Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-voice-detector.git
cd deepfake-voice-detector

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Prepare Dataset (1 minute)

Create the data directory structure:

```bash
data/
├── real/     # Put real voice audio files here
└── fake/     # Put AI-generated voice files here
```

**Minimum requirement**: 50+ audio files in each directory

### Quick Dataset Sources:
- **Real voices**: Record yourself, use LibriSpeech, Common Voice
- **Fake voices**: Generate with Google TTS, ElevenLabs, or download from ASVspoof

## 🎓 Train Model (1-30 minutes depending on dataset size)

```bash
python train.py
```

This will:
- Load your audio files
- Extract features
- Train the CNN model
- Save to `models/deepfake_detector.h5`
- Generate performance plots in `plots/`

**Expected output:**
```
Training samples: 160
Testing samples: 40
Epoch 1/50
5/5 [==============================] - 2s 400ms/step - loss: 0.6931 - accuracy: 0.5000
...
Model saved to models/deepfake_detector.h5
```

## 🌐 Run Web App (30 seconds)

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

### Using the Web App:
1. Click **"Browse files"** or drag & drop an audio file
2. View the **waveform** and **spectrogram**
3. See the **prediction** (Real or Fake)
4. Check the **confidence score**

## 📝 Programmatic Usage

```python
from src.model import load_saved_model
from src.preprocess import preprocess_audio
from src.features import extract_mel_spectrogram, prepare_cnn_input

# Load model
model = load_saved_model('models/deepfake_detector.h5')

# Preprocess audio
audio = preprocess_audio('my_audio.wav', sr=16000, duration=3.0)

# Extract features
features = extract_mel_spectrogram(audio, sr=16000)
features = prepare_cnn_input(features)

# Predict
prediction = model.predict(features)
print(f"Prediction: {'Fake' if prediction > 0.5 else 'Real'}")
```

## 🎯 Testing Your Model

Try with a test file:

```bash
python example_usage.py
```

Choose option 1 and enter the path to your audio file.

## 📈 Expected Results

On a balanced dataset of 200+ samples:
- **Accuracy**: 85-95%
- **Training time**: 5-30 minutes (CPU) or 2-10 minutes (GPU)
- **Inference time**: <1 second per audio file

## 🔧 Common Issues

### Issue: "No audio files found"
**Solution**: Make sure audio files are in `data/real/` and `data/fake/`

### Issue: "Model not found"
**Solution**: Run `python train.py` first

### Issue: Low accuracy (<70%)
**Solutions**:
- Add more training data (aim for 500+ samples)
- Ensure data quality (clear audio, proper labeling)
- Train for more epochs
- Try different feature types (MFCC vs Mel)

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the code in `src/` directory
- Experiment with different model architectures in `src/model.py`
- Try batch prediction with `example_usage.py`
- Customize training parameters in `train.py`

## 🆘 Need Help?

- Check [README.md](README.md) for detailed documentation
- Review code comments and docstrings
- Open an issue on GitHub
- Check [Troubleshooting](README.md#troubleshooting) section

---

**Ready to detect deepfakes? Let's go! 🚀**
