import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from src.inference.category import get_aqi_category


class AQIPredictor:
    def __init__(self,
                 model_path: str = "models/best_model.pkl",
                 preprocessor_path: str = "models/preprocessor.pkl",
                 metadata_path: str = "models/model_metadata.json"):
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = self._resolve_path(model_path, project_root)
        self.preprocessor_path = self._resolve_path(preprocessor_path, project_root)
        self.metadata_path = self._resolve_path(metadata_path, project_root)
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.load_artifacts()

    @staticmethod
    def _resolve_path(path: str, project_root: Path) -> str:
        artifact_path = Path(path)
        if artifact_path.is_absolute():
            return str(artifact_path)
        return str(project_root / artifact_path)

    def load_artifacts(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        if os.path.exists(self.preprocessor_path):
            with open(self.preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "best_model_name": "StandardModel",
                "use_scaled_features": False,
                "feature_order": ["pm2_5", "pm10", "no2", "so2", "co", "temperature", "humidity", "wind_speed"]
            }

    def predict(self, input_data: dict) -> dict:
        expected_features = self.metadata.get(
            "feature_order",
            ["pm2_5", "pm10", "no2", "so2", "co", "temperature", "humidity", "wind_speed"]
        )

        df = pd.DataFrame([input_data])[expected_features]

        if self.metadata.get("use_scaled_features", False):
            if self.preprocessor is None:
                raise RuntimeError("Preprocessor required but not loaded.")
            X = self.preprocessor.transform(df)
        else:
            X = df

        pred_val = float(self.model.predict(X)[0])
        pred_val = max(0.0, round(pred_val, 2))
        category = get_aqi_category(pred_val)

        return {
            "predicted_aqi": pred_val,
            "category": category,
            "model_version": self.metadata.get("best_model_name", "1.0.0")
        }