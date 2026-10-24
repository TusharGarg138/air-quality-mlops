import os
import pickle
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def run_preprocessing(config_path: str = "configs/params.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["data_source"]["raw_path"]
    target_col = config["base"]["target_col"]
    features = config["features"]["pollutants"] + config["features"]["weather"]
    random_state = config["base"]["random_state"]

    df = pd.read_csv(raw_path)
    X = df[features]
    y = df[target_col]

    test_size = config["preprocess"]["test_size"]
    val_size = config["preprocess"]["val_size"]

    # Split: Train + Val vs Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split: Train vs Val
    adj_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adj_val_size, random_state=random_state
    )

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Fit pipeline strictly on training split
    preprocessor.fit(X_train)

    os.makedirs(os.path.dirname(config["preprocess"]["preprocessor_path"]), exist_ok=True)
    with open(config["preprocess"]["preprocessor_path"], "wb") as f:
        pickle.dump(preprocessor, f)

    os.makedirs("data/processed", exist_ok=True)
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_parquet(config["preprocess"]["processed_train_path"], index=False)
    val_df.to_parquet(config["preprocess"]["processed_val_path"], index=False)
    test_df.to_parquet(config["preprocess"]["processed_test_path"], index=False)

    print("Preprocessing completed successfully. Artifacts saved.")


if __name__ == "__main__":
    run_preprocessing()