import json
import os
import yaml
import pandas as pd


def validate_raw_data(config_path: str = "configs/params.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["data_source"]["raw_path"]
    df = pd.read_csv(raw_path)

    expected_features = config["features"]["pollutants"] + config["features"]["weather"]
    target_col = config["base"]["target_col"]
    all_expected = expected_features + [target_col]

    missing_cols = [col for col in all_expected if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    null_counts = df[all_expected].isnull().sum().to_dict()
    duplicates = int(df.duplicated().sum())

    range_checks = {
        "pm2_5_min": float(df["pm2_5"].min()),
        "pm10_min": float(df["pm10"].min()),
        "humidity_max": float(df["humidity"].max()),
        "aqi_min": float(df["aqi"].min()),
        "aqi_max": float(df["aqi"].max()),
    }

    report = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "null_counts": null_counts,
        "duplicate_rows": duplicates,
        "range_checks": range_checks,
        "status": "PASSED" if duplicates == 0 and sum(null_counts.values()) == 0 else "WARNINGS_FOUND"
    }

    os.makedirs("reports", exist_ok=True)
    report_file = "reports/data_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validation successful. Report saved to {report_file}")


if __name__ == "__main__":
    validate_raw_data()