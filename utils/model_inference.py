# utils/model_inference.py
"""
Model loading, inference, and explainability using SHAP.
"""

import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "../model/deepfake_detector.pkl"
SCALER_PATH = "../model/scaler.pkl"

# -----------------------------
# Model & SHAP Explainer
# -----------------------------
class DeepfakeDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.explainer = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        """Load trained model and scaler."""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            print("✅ Model and scaler loaded.")

            # Initialize SHAP explainer (TreeExplainer for XGBoost)
            self.explainer = shap.TreeExplainer(self.model)
            print("✅ SHAP TreeExplainer initialized.")

            # Load feature names
            from utils.audio_processing import get_feature_names
            self.feature_names = get_feature_names()
            print(f"✅ Loaded {len(self.feature_names)} feature names.")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")

    def predict(self, features):
        """
        Predict if audio is real or fake.
        Args:
            features: np.array of shape (1, n_features)
        Returns:
            prediction, confidence, shap_values
        """
        if self.model is None or self.scaler is None:
            return "Error", 0.0, None

        try:
            scaled_features = self.scaler.transform(features)

            pred = self.model.predict(scaled_features)[0]
            prob = self.model.predict_proba(scaled_features)[0]
            confidence = float(np.max(prob))

            # Compute SHAP values
            shap_values = self.explainer.shap_values(scaled_features)

            label = "Fake (AI-Generated)" if pred == 1 else "Real (Human)"
            return label, confidence, shap_values

        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0, None

    def plot_shap_explanation(self, shap_values, features, max_display=10):
        """Create SHAP force plot as image."""
        try:
            # Summarize SHAP values for force plot
            shap_value = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
            feature_vec = features[0]

            fig, ax = plt.subplots(figsize=(10, 4))
            shap.force_plot(
                self.explainer.expected_value,
                shap_value,
                feature_vec,
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.2)

            # Save to bytes
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close(fig)

            return f'<img src="data:image/png;base64,{img_str}" style="width:100%">'
        except Exception as e:
            print(f"SHAP plot failed: {e}")
            return "<p>SHAP explanation not available.</p>"