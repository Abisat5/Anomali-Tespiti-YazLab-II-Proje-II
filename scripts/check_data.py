"""
data/raw altindaki SKAB ve BATADAL dosyalarini kontrol eder.
Kullanim: python scripts/check_data.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessing.data_loader import DataLoader


def main():
    config_path = os.path.join(ROOT, "configs", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    loader = DataLoader(config)
    print(f"Veri dizini: {loader.raw_path}\n")

    ok = True
    for dataset_name in config["project"]["datasets"]:
        print(f"--- {dataset_name} ---")
        df = loader.load_data(dataset_name)
        if df is None:
            ok = False
            print(f"[EKSIK] {dataset_name} yuklenemedi.\n")
            continue

        label_col = loader.detect_label_column(df, dataset_name)
        feature_cols = loader.get_feature_columns(df, label_col)
        print(f"Etiket sutunu: {label_col}")
        print(f"Ozellik sayisi: {len(feature_cols)}")
        print(f"Satir sayisi: {len(df)}")

        if dataset_name == "SKAB":
            groups = df.groupby("source_group")["source_file"].nunique()
            for group, count in groups.items():
                print(f"  {group}: {count} dosya")
        print()

    if ok:
        print("Tum veri setleri hazir. Calistirmak icin: python main.py")
    else:
        print("Eksik veri var. Uretmek icin: python scripts/generate_sample_data.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
