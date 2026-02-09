from typing import Any, Dict
import joblib
import numpy as np

class WalletAnalyzer:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def analyze_wallet(self, features: Dict[str, Any]) -> Dict[str, Any]:
        feature_array = self._prepare_features(features)
        prediction = self.model.predict(feature_array)
        score = self.model.predict_proba(feature_array)[0]
        return {
            "label": prediction[0],
            "score": score.tolist()  # Convert to list for JSON serialization
        }

    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        # Convert features to the format expected by the model
        return np.array([[features['feature1'], features['feature2'], features['feature3']]])  # Adjust based on actual features

def create_analyzer(model_path: str) -> WalletAnalyzer:
    return WalletAnalyzer(model_path)