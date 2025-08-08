# scripts/generate_demo_audio.py (updated)
import numpy as np
import soundfile as sf
import os

REAL_DIR = "../data/real"
FAKE_DIR = "../data/fake"
SAMPLE_RATE = 22050
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

def generate_human_like_voice(duration, sr):
    t = np.linspace(0, duration, int(duration * sr))
    base_freq = np.random.uniform(120, 220)
    harmonics = sum([0.6 / (i + 1) * np.sin(2 * np.pi * (i + 1) * base_freq * t) for i in range(6)])
    attack = np.linspace(0, 1, int(0.1 * sr))
    sustain = np.ones(len(t) - int(0.2 * sr))
    release = np.linspace(1, 0, int(0.1 * sr))
    envelope = np.concatenate([attack, sustain, release])[:len(t)]
    jitter = 0.01 * np.random.randn(len(t))
    y = (harmonics + jitter) * envelope * 0.3
    return np.clip(y, -1.0, 1.0)

def generate_ai_like_voice(duration, sr):
    t = np.linspace(0, duration, int(duration * sr))
    base_freq = 160
    harmonics = sum([0.4 / (i + 1) * np.sin(2 * np.pi * (i + 1) * base_freq * t) for i in range(4)])
    envelope = np.ones(len(t)) * 0.8
    glitch_interval = int(sr / 10)
    harmonics[::glitch_interval] += np.random.normal(0, 0.01, len(harmonics[::glitch_interval]))
    y = harmonics * envelope * 0.3
    return np.clip(y, -1.0, 1.0)

def generate_demo_files(n_samples=5):
    """Generate synthetic dataset in data/real and data/fake."""
    # Clear old files
    for folder in [REAL_DIR, FAKE_DIR]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

    # Generate real
    for i in range(1, n_samples + 1):
        y = generate_human_like_voice(np.random.uniform(2.5, 4.0), SAMPLE_RATE)
        sf.write(os.path.join(REAL_DIR, f"real_{i:03d}.wav"), y, SAMPLE_RATE)

    # Generate fake
    for i in range(1, n_samples + 1):
        y = generate_ai_like_voice(np.random.uniform(2.5, 4.0), SAMPLE_RATE)
        sf.write(os.path.join(FAKE_DIR, f"fake_{i:03d}.wav"), y, SAMPLE_RATE)

    print(f"✅ Generated {n_samples} real + {n_samples} fake samples")