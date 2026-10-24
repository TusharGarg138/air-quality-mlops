import json
import os
import pickle
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_final_model(config_path: str = "configs/params.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    target_col = config["base"]["target_col"]
    features = config["features"]["pollutants"] + config["features"]["weather"]

    test_df = pd.read_parquet(config["preprocess"]["processed_test_path"])
    X_test_raw = test_df[features]
    y_test = test_df[target_col]

    with open(config["train"]["selected_model_path"], "rb") as f:
        model = pickle.load(f)

    with open(config["train"]["model_metadata_path"], "r") as f:
        metadata = json.load(f)

    if metadata["use_scaled_features"]:
        with open(config["preprocess"]["preprocessor_path"], "rb") as f:
            preprocessor = pickle.load(f)
        X_test = preprocessor.transform(X_test_raw)
    else:
        X_test = X_test_raw

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    os.makedirs(os.path.dirname(config["evaluate"]["metrics_path"]), exist_ok=True)
    os.makedirs(config["evaluate"]["figures_dir"], exist_ok=True)

    metrics = {
        "test_mae": round(mae, 3),
        "test_rmse": round(rmse, 3),
        "test_r2": round(r2, 3),
        "evaluated_model": metadata["best_model_name"]
    }

    with open(config["evaluate"]["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate Actual vs Predicted Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, preds, alpha=0.4, color="#0d9488")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title(f"Actual vs Predicted AQI ({metadata['best_model_name']})")
    plt.tight_layout()
    plt.savefig(os.path.join(config["evaluate"]["figures_dir"], "actual_vs_predicted.png"))
    plt.close()

    # Generate Residuals Plot
    residuals = y_test - preds
    plt.figure(figsize=(7, 6))
    plt.scatter(preds, residuals, alpha=0.4, color="#3b82f6")
    plt.axhline(0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted AQI")
    plt.ylabel("Residuals")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(config["evaluate"]["figures_dir"], "residuals.png"))
    plt.close()

    print(f"Final Test Evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")


if __name__ == "__main__":
    evaluate_final_model()