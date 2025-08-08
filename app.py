# app.py
# VoiceGuard AI: Deepfake Audio Detector (Enhanced)
# Now with training, animation, and real/fake folder support
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import tempfile
from datetime import datetime
import base64

# Import mic recorder
from streamlit_mic_recorder import mic_recorder

# -----------------------------
# Configuration & Constants
# -----------------------------
st.set_page_config(
    page_title="🔊 VoiceGuard AI",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODEL_PATH = "model/deepfake_detector.pkl"
SCALER_PATH = "model/scaler.pkl"
SAMPLE_RATE = 22050
N_MFCC = 13
DATA_DIR = "data"
REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")

# Ensure directories
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

# Session state
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'anim_running' not in st.session_state:
    st.session_state.anim_running = False

# -----------------------------
# Utility Functions
# -----------------------------
def extract_mfcc(audio_path, n_mfcc=N_MFCC):
    """Extract MFCC features."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1).reshape(1, -1)
    except Exception as e:
        st.error(f"MFCC extraction failed: {e}")
        return None

def load_model_and_scaler():
    """Load model and scaler."""
    try:
        if not os.path.exists(MODEL_PATH):
            st.warning(f"Model not found: {MODEL_PATH}")
            return None, None
        if not os.path.exists(SCALER_PATH):
            st.warning(f"Scaler not found: {SCALER_PATH}")
            return None, None
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_deepfake(mfcc, model, scaler):
    """Predict if audio is real or fake."""
    try:
        mfcc_scaled = scaler.transform(mfcc)
        pred = model.predict(mfcc_scaled)[0]
        prob = model.predict_proba(mfcc_scaled)[0]
        conf = float(np.max(prob))
        return ("Fake (AI-Generated)" if pred == 1 else "Real (Human)"), conf
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error", 0.0

def save_uploadedfile(uploadedfile):
    """Save uploaded file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploadedfile.getbuffer())
            return tmp.name
    except Exception as e:
        st.error(f"Save failed: {e}")
        return None

def plot_waveform(y, sr, title="Waveform"):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig

def plot_spectrogram(y, sr, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title)
    plt.colorbar(ax.images[0], format="%+2.f dB")
    return fig

def add_to_history(filename, result, conf):
    st.session_state.detection_history.append({
        "filename": filename,
        "result": result,
        "confidence": f"{conf:.2%}",
        "timestamp": datetime.now().strftime("%H:%M")
    })

def display_history():
    st.sidebar.subheader("🔍 Detection History")
    if st.session_state.detection_history:
        for rec in reversed(st.session_state.detection_history):
            with st.sidebar.expander(f"{rec['filename']} ({rec['timestamp']})"):
                st.write(f"**Result**: {rec['result']}")
                st.write(f"**Confidence**: {rec['confidence']}")
    else:
        st.sidebar.info("No detections yet.")

def generate_sample_audio():
    """Generate demo audio if folders are empty."""
    if os.listdir(REAL_DIR) or os.listdir(FAKE_DIR):
        return False

    st.info("No audio found. Generating synthetic real/fake samples...")
    import numpy as np
    import soundfile as sf

    def synth_voice(duration, is_fake=False):
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        freq = 150
        harmonics = sum([0.5/(i+1) * np.sin(2*np.pi*(i+1)*freq*t) for i in range(5)])
        if is_fake:
            y = harmonics * 0.3
            y[::2000] += np.random.randn(len(y[::2000])) * 0.01
        else:
            env = np.hanning(len(t)) * 0.5 + 0.5
            y = harmonics * env * 0.3
        return y

    for i in range(3):
        y = synth_voice(3.0)
        sf.write(os.path.join(REAL_DIR, f"real_{i:03d}.wav"), y, SAMPLE_RATE)
        y = synth_voice(3.0, is_fake=True)
        sf.write(os.path.join(FAKE_DIR, f"fake_{i:03d}.wav"), y, SAMPLE_RATE)
    st.success("✅ Generated demo audio files.")
    return True

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🔊 VoiceGuard AI")
st.sidebar.markdown("> *Detecting AI voices, protecting identity.*")
st.sidebar.info("Detects synthetic voices using ML analysis.")

# Action selector
action = st.sidebar.radio("Choose Action", [
    "🎤 Record & Detect",
    "📤 Upload & Analyze",
    "🧪 Voice Clone Demo",
    "🔁 Train Model",
    "🎥 Live Spectrogram"
])

display_history()

st.sidebar.markdown("---")
st.sidebar.warning("""
⚠️ **Ethical Use Only**  
For education and research.  
Do not misuse.
""")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Record", "Upload", "Demo", "Train", "Live View"
])

# -----------------------------
# Tab 1: Record & Detect
# -----------------------------
with tab1:
    st.header("🎙️ Record Your Voice")
    st.info("Click to record from your microphone.")
    audio = mic_recorder(start_prompt="🟢 Start", stop_prompt="🔴 Stop", key="rec")

    if audio:
        st.audio(audio['bytes'], format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio['bytes'])
            temp_path = tmp.name

        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_waveform(y, sr, "Recorded Waveform"))
        with col2:
            st.pyplot(plot_spectrogram(y, sr, "Mel Spectrogram"))

        st.subheader("🔍 Result")
        with st.spinner("Analyzing..."):
            mfcc = extract_mfcc(temp_path)
            model, scaler = load_model_and_scaler()
            if mfcc is not None and model and scaler:
                result, conf = predict_deepfake(mfcc, model, scaler)
                st.success(f"**{result}**")
                st.metric("Confidence", f"{conf:.2%}")
                add_to_history("recorded.wav", result, conf)
            else:
                st.warning("Using dummy result (demo mode)")
                st.info("Prediction: Real (Human) | Confidence: 91%")

        os.unlink(temp_path)

# -----------------------------
# Tab 2: Upload & Analyze
# -----------------------------
with tab2:
    st.header("📁 Upload Audio")
    uploaded = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])
    if uploaded:
        st.audio(uploaded, format="audio/wav")
        temp_path = save_uploadedfile(uploaded)
        if temp_path:
            y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_waveform(y, sr, "Uploaded Waveform"))
            with c2:
                st.pyplot(plot_spectrogram(y, sr, "Mel Spectrogram"))

            st.subheader("🔍 Result")
            with st.spinner("Analyzing..."):
                mfcc = extract_mfcc(temp_path)
                model, scaler = load_model_and_scaler()
                if mfcc is not None and model and scaler:
                    result, conf = predict_deepfake(mfcc, model, scaler)
                    st.success(f"**{result}**")
                    st.metric("Confidence", f"{conf:.2%}")
                    add_to_history(uploaded.name, result, conf)
                else:
                    st.warning("Model not loaded.")
            os.unlink(temp_path)

# -----------------------------
# Tab 3: Voice Clone Demo
# -----------------------------
with tab3:
    st.header("🧪 AI Voice Cloning Demo")
    st.markdown("**For educational purposes only.**")

    # Generate sample data if needed
    generate_sample_audio()

    real_files = [f for f in os.listdir(REAL_DIR) if f.endswith(".wav")]
    fake_files = [f for f in os.listdir(FAKE_DIR) if f.endswith(".wav")]

    if real_files and fake_files:
        real_choice = real_files[0]
        fake_choice = fake_files[0]

        st.subheader("1. Real Human Voice")
        st.audio(os.path.join(REAL_DIR, real_choice))
        st.caption("Real person speaking.")

        st.subheader("2. AI-Generated Voice")
        st.audio(os.path.join(FAKE_DIR, fake_choice))
        st.caption("Synthetic voice (simulated).")

        st.info("🧠 Try uploading these in Tab 2!")

        st.subheader("3. Compare Waveforms")
        c1, c2 = st.columns(2)
        y1, sr1 = librosa.load(os.path.join(REAL_DIR, real_choice), sr=SAMPLE_RATE)
        y2, sr2 = librosa.load(os.path.join(FAKE_DIR, fake_choice), sr=SAMPLE_RATE)
        with c1:
            st.pyplot(plot_waveform(y1, sr1, "Real Voice"))
        with c2:
            st.pyplot(plot_waveform(y2, sr2, "Fake Voice"))
    else:
        st.warning("No demo audio available.")

# -----------------------------
# Tab 4: Train Model
# -----------------------------
with tab4:
    st.header("🔁 Retrain Detection Model")
    st.info("Regenerate data and retrain the XGBoost classifier.")

    if st.button("🗂️ Generate Sample Data"):
        generate_sample_audio()

    if st.button("🚀 Train Model"):
        with st.spinner("Training deepfake detector..."):
            try:
                from model.train_detector import train_model
                train_model()
                st.success("🎉 Model trained and saved!")
                st.balloons()
            except Exception as e:
                st.error(f"Training failed: {e}")

# -----------------------------
# Tab 5: Live Spectrogram Animation
# -----------------------------
with tab5:
    st.header("🎥 Real-Time Spectrogram Animation")
    buf = st.experimental_audio_input("🎤 Record for Animation")
    if buf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(buf.getvalue())
            temp_path = tmp.name

        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
        chunk_dur = 0.5
        chunk_size = int(chunk_dur * sr)
        chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_xlabel("Time"); ax.set_ylabel("Freq"); ax.set_ylim(0, 128)
        img = ax.imshow(np.zeros((128, 1)), aspect='auto', origin='lower', cmap='inferno')
        plt.tight_layout()
        frame_display = st.empty()
        plot_display = st.pyplot(fig)

        if st.button("▶️ Start Animation"):
            S_frames = []
            for i, chunk in enumerate(chunks):
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
                S_db = librosa.power_to_db(S, ref=np.max)
                S_frames.append(S_db)

                combined = np.concatenate(S_frames, axis=1)
                img.set_data(combined)
                img.set_clim(vmin=-80, vmax=0)
                frame_display.info(f"Frame {i+1}/{len(chunks)}")
                plot_display.pyplot(fig)

            st.success("🎬 Animation complete!")
        os.unlink(temp_path)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<center style='color:gray; font-size:0.9em'>"
    "VoiceGuard AI | Built with Streamlit & Librosa | For education only"
    "</center>",
    unsafe_allow_html=True
)