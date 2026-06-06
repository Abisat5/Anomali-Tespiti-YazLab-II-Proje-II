import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


class DataLoader:
    METADATA_COLS = {"datetime", "changepoint", "source_group", "source_file"}
    LABEL_CANDIDATES = ["anomaly", "attack", "att_flag", "label", "is_anomaly", "is_attack"]
    TIME_CANDIDATES = ["datetime", "timestamp", "date", "time", "datenum"]

    def __init__(self, config):
        self.config = config
        self.raw_path = config["paths"]["raw_data"]
        self.datasets = config["project"]["datasets"]
        self.skab_folders = config["paths"].get("skab_subfolders", ["valve1", "valve2"])

    def load_data(self, dataset_name):
        if dataset_name not in self.datasets:
            raise ValueError(f"[Hata] {dataset_name} config'de tanımlı değil.")

        print(f"\n[DataLoader] {dataset_name} veri seti yükleniyor...")
        if dataset_name == "SKAB":
            return self._load_skab()
        if dataset_name == "BATADAL":
            return self._load_batadal()
        return None

    def _load_skab(self):
        all_data = []
        skab_base_path = os.path.join(self.raw_path, "SKAB")

        for folder in self.skab_folders:
            folder_path = os.path.join(skab_base_path, folder)
            if not os.path.exists(folder_path):
                print(
                    f"[Uyarı] SKAB klasörü bulunamadı: {folder_path}. "
                    f"Lütfen data/raw/SKAB/{folder} dizinini oluşturun."
                )
                continue

            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            for file_path in csv_files:
                df = pd.read_csv(file_path, sep=";")
                df["source_group"] = folder
                df["source_file"] = os.path.basename(file_path)
                all_data.append(df)

        if not all_data:
            return None

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"[DataLoader] SKAB başarıyla birleştirildi. Toplam Boyut: {combined_df.shape}")
        return combined_df

    def _load_batadal(self):
        file_name = self.config["paths"].get("batadal_file", "batadal.csv")
        file_path = os.path.join(self.raw_path, file_name)
        try:
            df = pd.read_csv(file_path)
            print(f"[DataLoader] BATADAL yüklendi ({file_name}). Boyut: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"[Uyarı] {file_path} bulunamadı.")
            return None

    def detect_label_column(self, df, dataset_name):
        lower_map = {col.lower(): col for col in df.columns}

        if dataset_name == "SKAB" and "anomaly" in lower_map:
            return lower_map["anomaly"]

        for candidate in self.LABEL_CANDIDATES:
            if candidate in lower_map:
                return lower_map[candidate]

        bool_cols = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]
        if bool_cols:
            return bool_cols[-1]

        raise ValueError(f"[Hata] {dataset_name} için etiket sütunu tespit edilemedi.")

    def get_feature_columns(self, df, label_col):
        exclude = set(self.METADATA_COLS)
        exclude.add(label_col.lower())

        for col in df.columns:
            if col.lower() in self.TIME_CANDIDATES:
                exclude.add(col.lower())

        feature_cols = []
        for col in df.columns:
            if col.lower() in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

        if not feature_cols:
            raise ValueError("[Hata] Model girdisi için kullanılabilir sayısal sütun bulunamadı.")

        return feature_cols

    def handle_missing_values(self, df, feature_cols, label_col):
        cleaned = df.copy()
        subset = feature_cols + [label_col]
        before = len(cleaned)

        cleaned[subset] = cleaned[subset].replace([np.inf, -np.inf], np.nan)
        cleaned[subset] = cleaned[subset].ffill()
        cleaned = cleaned.dropna(subset=subset)

        print(
            f"[DataLoader] Eksik veri işlendi. "
            f"Kalan kayıt: {len(cleaned)} / {before}"
        )
        return cleaned.reset_index(drop=True)

    def split_data(self, df, dataset_name):
        if df is None:
            return None, None, None

        print(f"\n[DataLoader] {dataset_name} için veri bölme işlemi başlatılıyor...")

        if dataset_name == "BATADAL":
            train_ratio = self.config["data_split"]["train"]
            val_ratio = self.config["data_split"]["validation"]

            total_len = len(df)
            train_end = int(total_len * train_ratio)
            val_end = train_end + int(total_len * val_ratio)

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()

            print(
                f"[DataLoader - BATADAL] Kronolojik Split -> "
                f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
            )
            return train_df, val_df, test_df

        if dataset_name == "SKAB":
            return self._split_skab_by_files(df)

        return None, None, None

    def _split_skab_by_files(self, df):
        unique_files = np.array(sorted(df["source_file"].unique()))
        train_ratio = self.config["data_split"]["train"]
        val_ratio = self.config["data_split"]["validation"]

        n_train = int(len(unique_files) * train_ratio)
        n_val = int(len(unique_files) * val_ratio)

        train_files = unique_files[:n_train]
        val_files = unique_files[n_train:n_train + n_val]
        test_files = unique_files[n_train + n_val:]

        train_df = df[df["source_file"].isin(train_files)].copy()
        val_df = df[df["source_file"].isin(val_files)].copy()
        test_df = df[df["source_file"].isin(test_files)].copy()

        print(
            f"[DataLoader - SKAB] Dosya Bazlı Split -> "
            f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )
        return train_df, val_df, test_df

    def get_group_kfold_splits(self, df, label_col, n_splits=None):
        n_splits = n_splits or self.config["experiment_settings"].get("skab_cv_folds", 5)
        groups = df["source_file"].values
        y = df[label_col].values

        splitter_cls = StratifiedGroupKFold
        try:
            splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=42)
            split_iter = splitter.split(df, y, groups=groups)
        except Exception:
            print("[Uyarı] StratifiedGroupKFold kullanılamadı, GroupKFold'a düşülüyor.")
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(df, groups=groups)

        folds = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(split_iter, start=1):
            fold_df = df.iloc[train_val_idx].copy()
            test_df = df.iloc[test_idx].copy()

            unique_files = np.array(sorted(fold_df["source_file"].unique()))
            n_train_files = max(1, int(len(unique_files) * 0.75))
            train_files = unique_files[:n_train_files]
            val_files = unique_files[n_train_files:]

            if len(val_files) == 0 and len(train_files) > 1:
                val_files = train_files[-1:]
                train_files = train_files[:-1]

            train_df = fold_df[fold_df["source_file"].isin(train_files)].copy()
            val_df = fold_df[fold_df["source_file"].isin(val_files)].copy()

            folds.append(
                {
                    "fold": fold_idx,
                    "train": train_df.reset_index(drop=True),
                    "val": val_df.reset_index(drop=True),
                    "test": test_df.reset_index(drop=True),
                }
            )

        print(f"[DataLoader - SKAB] GroupKFold -> {len(folds)} fold oluşturuldu.")
        return folds
