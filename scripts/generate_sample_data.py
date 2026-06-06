"""
Hızlı pipeline testi için sentetik SKAB ve BATADAL verisi üretir.
Kullanım: python scripts/generate_sample_data.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")


def generate_skab():
    rng = np.random.default_rng(42)
    base = os.path.join(RAW, "SKAB")

    for group in ("valve1", "valve2"):
        group_path = os.path.join(base, group)
        os.makedirs(group_path, exist_ok=True)

        for idx in range(3):
            n = 300
            t = np.arange(n)
            sensor_a = np.sin(t / 10) + rng.normal(0, 0.05, n)
            sensor_b = np.cos(t / 8) + rng.normal(0, 0.05, n)
            sensor_c = rng.normal(0, 1, n)
            anomaly = np.zeros(n, dtype=int)
            anomaly[200:220] = 1
            if idx == 1:
                anomaly[80:95] = 1

            df = pd.DataFrame({
                "datetime": pd.date_range("2020-01-01", periods=n, freq="min"),
                "sensor_a": sensor_a,
                "sensor_b": sensor_b,
                "sensor_c": sensor_c,
                "changepoint": 0,
                "anomaly": anomaly,
            })
            df.loc[anomaly == 1, "sensor_c"] += 3.0
            out = os.path.join(group_path, f"sample_{group}_{idx}.csv")
            df.to_csv(out, sep=";", index=False)
            print(f"[OK] {out}")


def generate_batadal():
    os.makedirs(RAW, exist_ok=True)
    rng = np.random.default_rng(123)
    n = 1000
    t = np.arange(n)

    df = pd.DataFrame({
        "DATETIME": pd.date_range("2016-01-01", periods=n, freq="min"),
        "L_T1": 1.0 + 0.01 * np.sin(t / 15) + rng.normal(0, 0.02, n),
        "L_T2": 1.2 + 0.01 * np.cos(t / 12) + rng.normal(0, 0.02, n),
        "F1": 10 + rng.normal(0, 0.5, n),
        "S1": 5 + rng.normal(0, 0.3, n),
        "ATT_FLAG": 0,
    })
    df.loc[700:760, "ATT_FLAG"] = 1
    df.loc[700:760, "F1"] += 4.0

    out = os.path.join(RAW, "batadal.csv")
    df.to_csv(out, index=False)
    print(f"[OK] {out} (etiket sütunu: ATT_FLAG)")


if __name__ == "__main__":
    generate_skab()
    generate_batadal()
    print("\nSentetik veri üretildi. Şimdi çalıştır: python main.py")
