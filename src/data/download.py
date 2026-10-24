import os
import yaml
import numpy as np
import pandas as pd


def generate_cpcb_ground_truth(df: pd.DataFrame) -> pd.Series:
    """
    Computes Indian Central Pollution Control Board (CPCB) AQI sub-indices
    for PM2.5, PM10, NO2, SO2, and CO, taking the maximum sub-index as AQI.
    """
    def sub_index(val, breakpoints):
        for b_low, b_high, i_low, i_high in breakpoints:
            if b_low <= val <= b_high:
                return ((i_high - i_low) / (b_high - b_low)) * (val - b_low) + i_low
        return ((500 - 401) / (500 - 400)) * (val - 400) + 401 if val > 400 else 0

    pm25_bp = [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
               (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)]
    pm10_bp = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
               (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)]
    no2_bp = [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
              (181, 280, 201, 300), (281, 400, 301, 400), (401, 800, 401, 500)]
    so2_bp = [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
              (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)]
    co_bp = [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10.0, 101, 200),
             (10.1, 17.0, 201, 300), (17.1, 34.0, 301, 400), (34.1, 50.0, 401, 500)]

    s_pm25 = df["pm2_5"].apply(lambda x: sub_index(x, pm25_bp))
    s_pm10 = df["pm10"].apply(lambda x: sub_index(x, pm10_bp))
    s_no2 = df["no2"].apply(lambda x: sub_index(x, no2_bp))
    s_so2 = df["so2"].apply(lambda x: sub_index(x, so2_bp))
    s_co = df["co"].apply(lambda x: sub_index(x, co_bp))

    return np.round(pd.concat([s_pm25, s_pm10, s_no2, s_so2, s_co], axis=1).max(axis=1), 2)


def create_reproducible_dataset(config_path: str = "configs/params.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    n_samples = config["data_source"]["n_samples"]
    seed = config["base"]["random_state"]
    out_path = config["data_source"]["raw_path"]

    np.random.seed(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pm2_5 = np.random.gamma(shape=3.0, scale=18.0, size=n_samples)
    pm10 = pm2_5 * np.random.uniform(1.3, 2.1, size=n_samples) + np.random.normal(0, 5, n_samples)
    no2 = np.random.gamma(shape=2.5, scale=12.0, size=n_samples)
    so2 = np.random.gamma(shape=1.8, scale=6.0, size=n_samples)
    co = np.random.uniform(0.2, 3.5, size=n_samples)
    temperature = np.random.normal(27.5, 6.0, size=n_samples)
    humidity = np.clip(np.random.normal(60.0, 18.0, size=n_samples), 10.0, 100.0)
    wind_speed = np.clip(np.random.rayleigh(scale=3.0, size=n_samples), 0.2, 18.0)

    df = pd.DataFrame({
        "pm2_5": np.round(np.clip(pm2_5, 1.0, 480.0), 2),
        "pm10": np.round(np.clip(pm10, 2.0, 580.0), 2),
        "no2": np.round(np.clip(no2, 1.0, 350.0), 2),
        "so2": np.round(np.clip(so2, 0.5, 300.0), 2),
        "co": np.round(np.clip(co, 0.1, 15.0), 2),
        "temperature": np.round(temperature, 1),
        "humidity": np.round(humidity, 1),
        "wind_speed": np.round(wind_speed, 2),
    })

    df["aqi"] = generate_cpcb_ground_truth(df)
    df.to_csv(out_path, index=False)
    print(f"Generated raw dataset at {out_path} with {len(df)} records.")


if __name__ == "__main__":
    create_reproducible_dataset()