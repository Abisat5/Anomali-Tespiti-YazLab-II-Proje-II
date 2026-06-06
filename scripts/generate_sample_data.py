"""
Pipeline testi icin ornek SKAB ve BATADAL verisi uretir.
Kullanim: python scripts/generate_sample_data.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
SKAB_FILES_PER_VALVE = 5
SKAB_ROWS = 400
BATADAL_ROWS = 2000


def generate_skab():
    rng = np.random.default_rng(42)
    base = os.path.join(RAW, "SKAB")

    for group in ("valve1", "valve2"):
        group_path = os.path.join(base, group)
        os.makedirs(group_path, exist_ok=True)

        for idx in range(SKAB_FILES_PER_VALVE):
            n = SKAB_ROWS
            t = np.arange(n)
            phase = idx * 0.7
            sensor_a = np.sin(t / 10 + phase) + rng.normal(0, 0.05, n)
            sensor_b = np.cos(t / 8 + phase) + rng.normal(0, 0.05, n)
            sensor_c = rng.normal(0, 1, n)
            anomaly = np.zeros(n, dtype=int)

            start = 180 + idx * 15
            end = min(start + 20, n - 1)
            anomaly[start:end] = 1
            if idx % 2 == 1:
                anomaly[60:75] = 1

            df = pd.DataFrame({
                "datetime": pd.date_range("2020-01-01", periods=n, freq="min"),
                "sensor_a": sensor_a,
                "sensor_b": sensor_b,
                "sensor_c": sensor_c,
                "changepoint": 0,
                "anomaly": anomaly,
            })
            df.loc[anomaly == 1, "sensor_c"] += 3.0

            out = os.path.join(group_path, f"{group}_run_{idx:02d}.csv")
            df.to_csv(out, sep=";", index=False)
            print(f"[OK] {out}")


def generate_batadal():
    os.makedirs(RAW, exist_ok=True)
    batadal_dir = os.path.join(RAW, "BATADAL")
    os.makedirs(batadal_dir, exist_ok=True)

    rng = np.random.default_rng(123)
    n = BATADAL_ROWS
    t = np.arange(n)

    df = pd.DataFrame({
        "DATETIME": pd.date_range("2016-01-01", periods=n, freq="min"),
        "L_T1": 1.0 + 0.01 * np.sin(t / 15) + rng.normal(0, 0.02, n),
        "L_T2": 1.2 + 0.01 * np.cos(t / 12) + rng.normal(0, 0.02, n),
        "F1": 10 + rng.normal(0, 0.5, n),
        "S1": 5 + rng.normal(0, 0.3, n),
        "ATT_FLAG": 0,
    })
    df.loc[1400:1480, "ATT_FLAG"] = 1
    df.loc[1400:1480, "F1"] += 4.0

    outputs = [
        os.path.join(RAW, "batadal.csv"),
        os.path.join(batadal_dir, "Training Dataset 2.csv"),
    ]
    for out in outputs:
        df.to_csv(out, index=False)
        print(f"[OK] {out} (etiket sutunu: ATT_FLAG)")


if __name__ == "__main__":
    generate_skab()
    generate_batadal()
    print("\nOrnek veri uretildi.")
    print("Kontrol icin: python scripts/check_data.py")
    print("Calistirmak icin: python main.py")
