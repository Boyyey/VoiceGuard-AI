# utils/audio_processing.py
"""
Advanced audio feature extraction for deepfake detection.
Extracts MFCCs, spectral, and temporal features from audio files.
"""

import librosa
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Configuration
# -----------------------------
SAMPLE_RATE = 22050
N_MFCC = 13
MAX_DURATION = 5.0  # seconds
N_FFT = 2048
HOP_LENGTH = 512

# -----------------------------
# Feature Extraction
# -----------------------------
def load_audio(file_path, sr=SAMPLE_RATE, duration=None):
    """
    Load audio file with consistent settings.
    Returns: audio signal, sample rate
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        y, orig_sr = librosa.load(file_path, sr=None, duration=duration)
        
        # Resample if needed
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
            
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None


def pad_or_trim(y, sr, target_duration=MAX_DURATION):
    """Pad with zeros or trim to fixed duration."""
    target_samples = int(target_duration * sr)
    if len(y) > target_samples:
        return y[:target_samples]
    elif len(y) < target_samples:
        pad_len = target_samples - len(y)
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        return np.pad(y, (pad_left, pad_right), mode='constant')
    return y


def extract_mfcc_features(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Extract mean and variance of MFCCs."""
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return np.hstack([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
            np.mean(delta2, axis=1),
            np.std(delta2, axis=1)
        ])
    except Exception as e:
        print(f"MFCC extraction failed: {e}")
        return np.zeros((n_mfcc * 6,))


def extract_spectral_features(y, sr=SAMPLE_RATE):
    """Extract spectral features: centroid, rolloff, flatness, bandwidth."""
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=N_FFT, hop_length=HOP_LENGTH)
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        return np.hstack([
            np.mean(cent), np.std(cent),
            np.mean(rolloff), np.std(rolloff),
            np.mean(flatness), np.std(flatness),
            np.mean(bandwidth), np.std(bandwidth)
        ])
    except Exception as e:
        print(f"Spectral features failed: {e}")
        return np.zeros((8,))


def extract_temporal_features(y):
    """Extract zero-crossing rate and RMS energy."""
    try:
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        return np.hstack([
            np.mean(zcr), np.std(zcr),
            np.mean(rms), np.std(rms)
        ])
    except Exception as e:
        print(f"Temporal features failed: {e}")
        return np.zeros((4,))


def extract_chroma(y, sr=SAMPLE_RATE):
    """Extract chroma features for tonal analysis."""
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        return np.hstack([np.mean(chroma, axis=1), np.std(chroma, axis=1)])
    except Exception as e:
        print(f"Chroma features failed: {e}")
        return np.zeros((24,))  # 12 bins x 2 (mean, std)


def extract_all_features(file_path):
    """
    Extract full feature vector from audio file.
    Returns: 1D numpy array of shape (n_features,)
    """
    y, sr = load_audio(file_path, duration=MAX_DURATION)
    if y is None or len(y) == 0:
        return None

    y = pad_or_trim(y, sr)

    mfcc_feats = extract_mfcc_features(y, sr)
    spec_feats = extract_spectral_features(y, sr)
    temp_feats = extract_temporal_features(y)
    chroma_feats = extract_chroma(y, sr)

    feature_vector = np.hstack([
        mfcc_feats,
        spec_feats,
        temp_feats,
        chroma_feats
    ])

    return feature_vector.reshape(1, -1)


def get_feature_names():
    """Return list of feature names for explainability."""
    mfcc_names = [f'MFCC_{i}_mean' for i in range(N_MFCC)] + \
                 [f'MFCC_{i}_std' for i in range(N_MFCC)] + \
                 [f'MFCC_delta_{i}_mean' for i in range(N_MFCC)] + \
                 [f'MFCC_delta_{i}_std' for i in range(N_MFCC)] + \
                 [f'MFCC_delta2_{i}_mean' for i in range(N_MFCC)] + \
                 [f'MFCC_delta2_{i}_std' for i in range(N_MFCC)]

    spec_names = [
        'Spectral_Centroid_mean', 'Spectral_Centroid_std',
        'Spectral_Rolloff_mean', 'Spectral_Rolloff_std',
        'Spectral_Flatness_mean', 'Spectral_Flatness_std',
        'Bandwidth_mean', 'Bandwidth_std'
    ]

    temp_names = [
        'ZCR_mean', 'ZCR_std',
        'RMS_mean', 'RMS_std'
    ]

    chroma_names = [f'Chroma_{i}_mean' for i in range(12)] + [f'Chroma_{i}_std' for i in range(12)]

    return mfcc_names + spec_names + temp_names + chroma_names