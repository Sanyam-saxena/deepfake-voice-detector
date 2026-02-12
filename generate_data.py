"""
Synthetic Audio Data Generator for Deepfake Voice Detection
Generates synthetic "real" and "fake" voice-like audio samples for training.

NOTE: These are synthetic approximations, not actual voice recordings.
For production use, replace with real deepfake datasets (e.g., ASVspoof).
"""

import os
import numpy as np
import soundfile as sf

# Configuration
SAMPLE_RATE = 16000
DURATION = 3.0  # seconds
NUM_SAMPLES = 10  # per class
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def generate_real_voice_sample(sample_rate=SAMPLE_RATE, duration=DURATION):
    """
    Generate a synthetic 'real' voice-like audio signal.
    Simulates natural human voice characteristics:
    - Multiple harmonics (fundamental + overtones)
    - Natural vibrato (pitch variation)
    - Amplitude envelope (attack, sustain, decay)
    - Slight breathiness (noise component)
    - Natural pitch drift
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Random fundamental frequency (human voice range)
    f0 = np.random.uniform(100, 300)
    
    # Natural vibrato (slight pitch wobble)
    vibrato_rate = np.random.uniform(4, 7)  # Hz
    vibrato_depth = np.random.uniform(2, 8)  # Hz variation
    pitch = f0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    
    # Natural pitch drift over time
    pitch += np.random.uniform(-5, 5) * t / duration
    
    # Generate phase from instantaneous frequency
    phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
    
    # Fundamental + harmonics with natural rolloff
    signal = np.sin(phase)  # fundamental
    signal += 0.5 * np.sin(2 * phase)  # 2nd harmonic
    signal += 0.25 * np.sin(3 * phase)  # 3rd harmonic
    signal += 0.12 * np.sin(4 * phase)  # 4th harmonic
    signal += 0.06 * np.sin(5 * phase)  # 5th harmonic
    
    # Natural amplitude envelope (attack-sustain-decay)
    attack = int(0.05 * sample_rate)
    decay = int(0.3 * sample_rate)
    envelope = np.ones(len(t))
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-decay:] = np.linspace(1, 0, decay)
    
    # Add slight amplitude modulation (natural tremor)
    tremor = 1 + 0.05 * np.sin(2 * np.pi * np.random.uniform(3, 6) * t)
    signal *= envelope * tremor
    
    # Add breathiness (filtered noise)
    breath = np.random.randn(len(t)) * 0.02
    signal += breath
    
    # Random pauses/syllable breaks
    num_breaks = np.random.randint(2, 5)
    for _ in range(num_breaks):
        break_start = np.random.randint(0, len(t) - int(0.15 * sample_rate))
        break_len = int(np.random.uniform(0.03, 0.1) * sample_rate)
        fade = np.linspace(1, 0.1, break_len)
        signal[break_start:break_start + break_len] *= fade
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    
    return signal.astype(np.float32)


def generate_fake_voice_sample(sample_rate=SAMPLE_RATE, duration=DURATION):
    """
    Generate a synthetic 'fake' (deepfake-like) audio signal.
    Simulates characteristics often found in AI-generated speech:
    - More uniform/robotic tone
    - Less natural variation
    - Subtle metallic artifacts
    - Periodic micro-glitches
    - Overly smooth amplitude
    - Unnatural harmonic structure
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Fixed fundamental (less natural variation)
    f0 = np.random.uniform(120, 250)
    
    # Very slight or no vibrato (robotic quality)
    vibrato_depth = np.random.uniform(0, 1)  # much less than real
    pitch = f0 + vibrato_depth * np.sin(2 * np.pi * 5 * t)
    
    phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
    
    # Harmonics with unnatural distribution
    signal = np.sin(phase)
    signal += 0.6 * np.sin(2 * phase)  # stronger 2nd harmonic
    signal += 0.4 * np.sin(3 * phase)  # unnaturally strong 3rd
    signal += 0.3 * np.sin(4 * phase)  # too much 4th
    signal += 0.2 * np.sin(5 * phase)  # too much 5th
    
    # Add metallic/digital artifacts (high-frequency components)
    artifact_freq = np.random.uniform(3000, 6000)
    signal += 0.03 * np.sin(2 * np.pi * artifact_freq * t)
    
    # Overly smooth envelope (AI-generated characteristic)
    envelope = np.ones(len(t))
    attack = int(0.01 * sample_rate)  # too sharp attack
    envelope[:attack] = np.linspace(0, 1, attack)
    signal *= envelope
    
    # Add periodic micro-glitches (synthesis artifacts)
    num_glitches = np.random.randint(3, 8)
    for _ in range(num_glitches):
        glitch_pos = np.random.randint(0, len(t) - 200)
        glitch_len = np.random.randint(50, 200)
        # Phase discontinuity
        signal[glitch_pos:glitch_pos + glitch_len] *= np.random.uniform(0.7, 1.3)
        # Small click artifact
        signal[glitch_pos] += np.random.uniform(-0.1, 0.1)
    
    # Quantization-like noise (digital artifact)
    quantization_noise = np.round(signal * 50) / 50 - signal
    signal += 0.1 * quantization_noise
    
    # Periodic buzz (vocoder artifact)
    buzz_freq = np.random.uniform(50, 100)
    signal += 0.015 * np.sign(np.sin(2 * np.pi * buzz_freq * t))
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    
    return signal.astype(np.float32)


def main():
    """Generate synthetic training data."""
    print("\n" + "=" * 60)
    print("SYNTHETIC AUDIO DATA GENERATOR")
    print("=" * 60)
    
    # Create directories
    real_dir = os.path.join(OUTPUT_DIR, 'real')
    fake_dir = os.path.join(OUTPUT_DIR, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots'), exist_ok=True)
    
    print(f"\nGenerating {NUM_SAMPLES} 'real' voice samples...")
    for i in range(NUM_SAMPLES):
        audio = generate_real_voice_sample()
        filepath = os.path.join(real_dir, f'real_voice_{i+1:03d}.wav')
        sf.write(filepath, audio, SAMPLE_RATE)
        print(f"  Created: {os.path.basename(filepath)} ({len(audio)/SAMPLE_RATE:.1f}s)")
    
    print(f"\nGenerating {NUM_SAMPLES} 'fake' voice samples...")
    for i in range(NUM_SAMPLES):
        audio = generate_fake_voice_sample()
        filepath = os.path.join(fake_dir, f'fake_voice_{i+1:03d}.wav')
        sf.write(filepath, audio, SAMPLE_RATE)
        print(f"  Created: {os.path.basename(filepath)} ({len(audio)/SAMPLE_RATE:.1f}s)")
    
    # Count files
    real_count = len([f for f in os.listdir(real_dir) if f.endswith('.wav')])
    fake_count = len([f for f in os.listdir(fake_dir) if f.endswith('.wav')])
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Real samples: {real_count} files in {real_dir}")
    print(f"  Fake samples: {fake_count} files in {fake_dir}")
    print(f"  Models dir:   created")
    print(f"  Plots dir:    created")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
