# model/train_detector.py
# Train a deepfake audio classifier using synthetic dataset
# Simulates real vs fake voice detection using MFCC features
# Generates and saves model + scaler

import numpy as np
import pandas as pd
import librosa
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "../data"  # Real and fake audio folders
REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")
SAMPLE_RATE = 22050
N_MFCC = 13
MAX_AUDIO_DURATION = 5.0  # seconds
N_SAMPLES = 200  # total samples (100 real, 100 fake) - simulate small dataset
OUTPUT_MODEL = "deepfake_detector.pkl"
OUTPUT_SCALER = "scaler.pkl"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(os.path.join(DATA_DIR, "")), exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

# -----------------------------
# Simulate Audio Dataset (if none exists)
# -----------------------------
def generate_silence_padded_audio(y, target_duration, sr):
    """Pad or trim audio to fixed duration."""
    target_samples = int(target_duration * sr)
    if len(y) > target_samples:
        return y[:target_samples]
    else:
        pad_len = target_samples - len(y)
        return np.pad(y, (0, pad_len), mode='constant')

def create_synthetic_real_voice(duration=3.0, sr=22050):
    """Generate a realistic but synthetic 'human' voice-like signal (sine + noise + prosody)."""
    t = np.linspace(0, duration, int(duration * sr))
    # Simulate voiced speech (vowels)
    base_freq = 150  # Fundamental frequency (male voice)
    harmonics = sum([0.5/((i+1)) * np.sin(2 * np.pi * (i+1) * base_freq * t) for i in range(5)])
    noise = 0.1 * np.random.normal(0, 1, len(t))
    envelope = np.hanning(int(sr * 0.1))  # Smooth on/off
    attack = np.concatenate([np.linspace(0, 1, 2205), np.ones(len(t) - 2205 - 2205), np.linspace(1, 0, 2205)])
    y = (harmonics + noise) * attack
    return y

def create_synthetic_fake_voice(duration=3.0, sr=22050):
    """Simulate AI-generated voice artifacts: less dynamic, flat prosody, spectral glitches."""
    t = np.linspace(0, duration, int(duration * sr))
    base_freq = 160
    harmonics = sum([0.5/((i+1)) * np.sin(2 * np.pi * (i+1) * base_freq * t) for i in range(5)])
    # Add spectral flatness, reduced dynamics
    noise = 0.05 * np.random.normal(0, 1, len(t))
    # Artificially smooth amplitude
    envelope = np.ones(len(t)) * 0.8
    y = (harmonics + noise) * envelope
    # Add tiny glitches
    glitch_pos = np.random.choice(len(y), 5)
    y[glitch_pos] += 0.02 * np.random.randn(5)
    return y

def generate_dataset():
    """Generate and save synthetic real/fake audio files."""
    print("🔍 Checking for dataset...")
    if os.path.exists(REAL_DIR) and len(os.listdir(REAL_DIR)) > 0:
        print("✅ Real audio found.")
    else:
        print("⚙️ Generating synthetic real voices...")
        for i in range(100):
            y = create_synthetic_real_voice(duration=np.random.uniform(2, 5))
            sf.write(os.path.join(REAL_DIR, f"real_{i:03d}.wav"), y, SAMPLE_RATE)
        print("✅ Generated 100 real samples.")

    if os.path.exists(FAKE_DIR) and len(os.listdir(FAKE_DIR)) > 0:
        print("✅ Fake audio found.")
    else:
        print("⚙️ Generating synthetic fake voices...")
        for i in range(100):
            y = create_synthetic_fake_voice(duration=np.random.uniform(2, 5))
            sf.write(os.path.join(FAKE_DIR, f"fake_{i:03d}.wav"), y, SAMPLE_RATE)
        print("✅ Generated 100 fake samples.")

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_audio_features(file_path, n_mfcc=13):
    """Extract robust features from audio."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = generate_silence_padded_audio(y, MAX_AUDIO_DURATION, sr)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Spectral features
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        flatness = librosa.feature.spectral_flatness(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)

        features = np.hstack([
            mfcc_mean,
            np.mean(cent),
            np.mean(rolloff),
            np.mean(flatness),
            np.mean(zcr),
            np.std(mfcc, axis=1),
            np.var(mfcc, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# -----------------------------
# Train Model
# -----------------------------
def train_model():
    print("🚀 Starting model training pipeline...")
    generate_dataset()

    features = []
    labels = []

    # Process real files
    for f in os.listdir(REAL_DIR):
        if f.endswith(".wav"):
            path = os.path.join(REAL_DIR, f)
            feat = extract_audio_features(path)
            if feat is not None:
                features.append(feat)
                labels.append(0)  # 0 = real

    # Process fake files
    for f in os.listdir(FAKE_DIR):
        if f.endswith(".wav"):
            path = os.path.join(FAKE_DIR, f)
            feat = extract_audio_features(path)
            if feat is not None:
                features.append(feat)
                labels.append(1)  # 1 = fake

    X = np.array(features)
    y = np.array(labels)
    X, y = shuffle(X, y, random_state=42)

    print(f"📊 Dataset shape: {X.shape}, Class distribution: Real={np.sum(y==0)}, Fake={np.sum(y==1)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("🤖 Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("✅ Training complete!")
    print("\n🔍 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    print(f"\n📈 ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plt.close()

    # Feature Importance
    importance = model.feature_importances_
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importance)[::-1]
    plt.title("Feature Importances")
    plt.bar(range(len(importance)), importance[indices], align="center")
    plt.xticks(range(len(importance)), [f"F{idx}" for idx in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"))
    plt.close()

    # Save model and scaler
    joblib.dump(model, OUTPUT_MODEL)
    joblib.dump(scaler, OUTPUT_SCALER)
    print(f"💾 Model saved: {OUTPUT_MODEL}")
    print(f"💾 Scaler saved: {OUTPUT_SCALER}")

    return model, scaler, X_test_scaled, y_test, y_pred

# -----------------------------
# Run Training
# -----------------------------
if __name__ == "__main__":
    import soundfile as sf  # Moved here to avoid top-level conflict
    model, scaler, X_test, y_true, y_pred = train_model()