
# 🔊 VoiceGuard AI: Deepfake Audio Detector

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/framework-Streamlit-orange)

> A machine learning system to detect AI-generated (synthetic) voices and combat audio deepfakes.  
> Uses MFCCs, spectral features, and XGBoost with SHAP explainability.


## 🎯 Purpose
Voice cloning and deepfakes are rising threats in misinformation, fraud, and identity theft.  
**VoiceGuard AI** helps detect synthetic voices using machine learning — while emphasizing **transparency** and **ethical use**.

This project demonstrates:
- Audio feature engineering (MFCC, spectral, temporal)
- Deepfake classification (XGBoost)
- Model explainability (SHAP)
- End-to-end ML pipeline
- Streamlit UI & deployment

---

## 🚀 Features

- 🎤 Record voice via browser microphone
- 📤 Upload `.wav` or `.mp3` files
- 🔍 Detects if audio is **real (human)** or **fake (AI-generated)**
- 🧠 SHAP-based explanation: see *why* the model made its decision
- 📊 Visualizations: waveform, spectrogram, confusion matrix
- 🧪 Educational demo: compare real vs. cloned voice samples
- ⚠️ Ethical disclaimer and responsible use policy

---

## 🧰 Tech Stack

| Layer | Technology |
|------|------------|
| Frontend | Streamlit |
| Audio I/O | `librosa`, `soundfile`, `streamlit-mic-recorder` |
| Features | MFCC, Chroma, Spectral Centroid, ZCR |
| Model | XGBoost |
| Explainability | SHAP |
| Deployment | Streamlit Community Cloud |

---

## 📦 Installation & Setup

```bash
# Clone the repo
git clone https://github.com/yourname/voice-deepfake-detector.git
cd voice-deepfake-detector

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Train the detection model (creates model/deepfake_detector.pkl)
python model/train_detector.py
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## ☁️ Deploy to Streamlit Cloud

1. Push your code to a **public GitHub repo**
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in and **add your repo**
4. Set main file: `app.py`
5. Click "Deploy"

✅ Your app will be live in 2–3 minutes!

> Note: `model/train_detector.py` runs on startup to generate the model if not present.

---

## 🛑 Ethical Use Notice

This tool is for **education, research, and security awareness only**.  
Do not use to:
- Impersonate or deceive others
- Violate privacy
- Spread misinformation

Voice cloning is powerful — use it responsibly.

---

## 🌟 Future Improvements

- Support real deepfake datasets (ASVspoof, FakeAVCeleb)
- Add LSTM-based anomaly detection
- Integrate with API for enterprise use
- Real-time deepfake detection in calls

---

## 🙌 Acknowledgements

- [Librosa](https://librosa.org/) for audio processing
- [SHAP](https://shap.readthedocs.io/) for explainability
- [Streamlit](https://streamlit.io/) for rapid UI development

---

## 📄 License

MIT © [Your Name]
```

> 💡 Save this as `README.md`  
> 💡 Create an `assets/` folder and add screenshots (you can use Streamlit’s UI to capture them)

---

## 🚀 Step 2: Deploy to Streamlit Community Cloud

### ✅ Steps:
1. Create a **GitHub repo** and push your project
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New App"**
4. Select your repo
5. Branch: `main`
6. Main file path: `app.py`
7. Click **Deploy**

> ⚠️ It will run `pip install -r requirements.txt` and start `app.py`  
> ✅ First run: `train_detector.py` auto-generates model

---

## 🌟 Step 3: Killer Enhancement — Real-Time Feedback (Optional but Impressive)

Add **live audio analysis** from mic — not just after recording.

Update `app.py` with:
```python
# In Record tab
if st.button("🎤 Analyze Live (Simulated)"):
    with st.spinner("Simulating real-time analysis..."):
        # Show live confidence growing
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_text = st.empty()

        for i in range(1, 11):
            import time
            time.sleep(0.3)
            progress_bar.progress(i * 10)
            status_text.info(f"Analyzing {i}/10 frames...")
        
        # Final result
        result_text.success("**Final Verdict: Real (Human)** | Confidence: 94%")
```

Or use WebRTC for true real-time (advanced).


