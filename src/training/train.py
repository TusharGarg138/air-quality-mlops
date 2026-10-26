import json
import os
import pickle
import time
from contextlib import nullcontext
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None


def train_models(config_path: str = "configs/params.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    target_col = config["base"]["target_col"]
    features = config["features"]["pollutants"] + config["features"]["weather"]
    seed = config["base"]["random_state"]

    train_df = pd.read_parquet(config["preprocess"]["processed_train_path"])
    val_df = pd.read_parquet(config["preprocess"]["processed_val_path"])

    with open(config["preprocess"]["preprocessor_path"], "rb") as f:
        preprocessor = pickle.load(f)

    X_train_raw = train_df[features]
    y_train = train_df[target_col]
    X_val_raw = val_df[features]
    y_val = val_df[target_col]

    X_train_scaled = preprocessor.transform(X_train_raw)
    X_val_scaled = preprocessor.transform(X_val_raw)

    candidates = {
        "LinearRegression": {
            "model": LinearRegression(**config["models"]["linear_regression"]),
            "use_scaled": True
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=seed, **config["models"]["random_forest"]),
            "use_scaled": False
        },
    }
    if XGBRegressor is not None:
        candidates["XGBoost"] = {
            "model": XGBRegressor(random_state=seed, **config["models"]["xgboost"]),
            "use_scaled": False
        }

    if mlflow is not None:
        mlflow.set_experiment("AQI_Prediction_Pipeline")
    results = {}

    for name, candidate in candidates.items():
        run_context = mlflow.start_run(run_name=name) if mlflow is not None else nullcontext()
        with run_context:
            start_time = time.time()
            model = candidate["model"]
            X_tr = X_train_scaled if candidate["use_scaled"] else X_train_raw
            X_v = X_val_scaled if candidate["use_scaled"] else X_val_raw

            model.fit(X_tr, y_train)
            duration = time.time() - start_time

            preds = model.predict(X_v)
            mae = float(mean_absolute_error(y_val, preds))
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            r2 = float(r2_score(y_val, preds))

            if mlflow is not None:
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({"val_mae": mae, "val_rmse": rmse, "val_r2": r2, "train_time_sec": duration})
                trusted_types = ["sklearn.tree._tree.Tree"]
                if name == "XGBoost":
                    trusted_types.extend([
                        "xgboost.core.Booster",
                        "xgboost.sklearn.XGBRegressor",
                    ])
                mlflow.sklearn.log_model(
                    model,
                    name=name,
                    skops_trusted_types=trusted_types
                )

            results[name] = {
                "model_obj": model,
                "use_scaled": candidate["use_scaled"],
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            print(f"[{name}] Val RMSE: {rmse:.3f} | Val MAE: {mae:.3f} | Val R²: {r2:.3f}")

    # Select best model based on minimum validation RMSE
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model_data = results[best_name]
    print(f"\nWinning Model: {best_name} with RMSE: {best_model_data['rmse']:.3f}")

    os.makedirs(os.path.dirname(config["train"]["selected_model_path"]), exist_ok=True)
    with open(config["train"]["selected_model_path"], "wb") as f:
        pickle.dump(best_model_data["model_obj"], f)

    metadata = {
        "best_model_name": best_name,
        "use_scaled_features": best_model_data["use_scaled"],
        "feature_order": features,
        "val_rmse": best_model_data["rmse"],
        "val_mae": best_model_data["mae"],
        "val_r2": best_model_data["r2"],
        "target": target_col,
        "standard": "CPCB India"
    }

    with open(config["train"]["model_metadata_path"], "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    train_models()