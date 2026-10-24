import os
import pickle
import numpy as np


def test_model_artifact_loadable():
    model_path = "models/best_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        dummy_input = np.ones((1, 8))
        preds = model.predict(dummy_input)
        assert len(preds) == 1
        assert np.isfinite(preds[0])