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
        self.raw_path = self._resolve_path(config["paths"]["raw_data"])
        self.datasets = config["project"]["datasets"]
        self.skab_folders = config["paths"].get("skab_subfolders", ["valve1", "valve2"])

    @staticmethod
    def _resolve_path(path):
        if os.path.isabs(path):
            return path
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return os.path.normpath(os.path.join(project_root, path))

    @staticmethod
    def _read_csv(file_path):
        for sep in (";", ","):
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
        return pd.read_csv(file_path)

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

            csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
            if not csv_files:
                print(f"[Uyarı] {folder_path} içinde .csv dosyası yok.")
                continue

            for file_path in csv_files:
                df = self._read_csv(file_path)
                df["source_group"] = folder
                df["source_file"] = os.path.basename(file_path)
                all_data.append(df)
                print(f"[DataLoader] SKAB dosyası yüklendi: {folder}/{os.path.basename(file_path)}")

        if not all_data:
            self._print_skab_setup_hint()
            return None

        combined_df = pd.concat(all_data, ignore_index=True)
        file_count = combined_df["source_file"].nunique()
        print(
            f"[DataLoader] SKAB birleştirildi. "
            f"Boyut: {combined_df.shape}, dosya sayısı: {file_count}"
        )
        return combined_df

    def _print_skab_setup_hint(self):
        print("[Hata] SKAB verisi bulunamadı.")
        print("Beklenen yapı:")
        for folder in self.skab_folders:
            print(f"  {os.path.join(self.raw_path, 'SKAB', folder)}/*.csv")
        print("Hızlı test için: python scripts/generate_sample_data.py")

    def _batadal_search_paths(self):
        configured = self.config["paths"].get("batadal_file", "batadal.csv")
        file_names = [
            configured,
            "Training Dataset 2.csv",
            "Training_Dataset_2.csv",
            "training_dataset_2.csv",
            "BATADAL_dataset03.csv",
            "BATADAL_dataset04.csv",
            "batadal.csv",
        ]
        seen = set()
        search_dirs = [
            self.raw_path,
            os.path.join(self.raw_path, "BATADAL"),
        ]
        paths = []
        for directory in search_dirs:
            for file_name in file_names:
                key = (directory, file_name)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(os.path.join(directory, file_name))
        return paths

    def _normalize_batadal_labels(self, df, label_col):
        """Training Dataset 2 kismi etiketli: -999 = etiketsiz/normal kabul edilir."""
        normalized = df.copy()
        normalized[label_col] = normalized[label_col].replace(-999, 0)
        normalized[label_col] = normalized[label_col].astype(int)

        counts = normalized[label_col].value_counts().to_dict()
        print(
            f"[DataLoader] BATADAL etiketleri normalize edildi (-999 -> 0). "
            f"Dagilim: {counts}"
        )
        return normalized

    def _validate_batadal_df(self, df, file_path):
        label_col = self.detect_label_column(df, "BATADAL")
        labels = set(df[label_col].dropna().unique())
        row_count = len(df)
        file_name = os.path.basename(file_path)

        if "ATT_FLAG" not in df.columns.str.strip().tolist() and label_col.lower() != "att_flag":
            print(f"[Hata] {file_name} etiket sutunu yok; bu Test Dataset olabilir.")
            return False

        if labels == {-999} or labels == {-999, 1} or -999 in labels:
            print(
                f"[DataLoader] {file_name} kismi etiketli Training Dataset 2 formatinda "
                f"(-999 + 1). Bu resmi indirmede normaldir."
            )

        if 1 not in labels:
            print(
                f"[Hata] {file_name} saldiri etiketi (1) icermiyor. "
                "Training Dataset 1 veya yanlis dosya olabilir."
            )
            return False

        if row_count < 2000:
            print(f"[Uyari] {file_name} satir sayisi beklenenden az ({row_count}).")
        return True

    def _load_batadal(self):
        for file_path in self._batadal_search_paths():
            if not os.path.exists(file_path):
                continue
            df = self._read_csv(file_path)
            df.columns = df.columns.str.strip()
            print(
                f"[DataLoader] BATADAL yüklendi: {file_path} "
                f"(Boyut: {df.shape})"
            )
            if not self._validate_batadal_df(df, file_path):
                continue
            label_col = self.detect_label_column(df, "BATADAL")
            df = self._normalize_batadal_labels(df, label_col)
            return df

        print("[Hata] BATADAL dosyası bulunamadı.")
        print("Beklenen konumlar (Training Dataset 2):")
        for file_path in self._batadal_search_paths():
            print(f"  {file_path}")
        print("Hızlı test için: python scripts/generate_sample_data.py")
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
        cv_seed = self.config["experiment_settings"].get("cv_random_state", 42)
        fold_train_ratio = self.config["data_split"].get("fold_train_ratio", 0.75)

        splitter_cls = StratifiedGroupKFold
        try:
            splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=cv_seed)
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
            n_train_files = max(1, int(len(unique_files) * fold_train_ratio))
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
