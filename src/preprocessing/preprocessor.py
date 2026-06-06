import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    EXCLUDE_KEYWORDS = {
        "label", "attack", "is_anomaly", "is_attack", "anomaly", "att_flag",
        "timestamp", "datetime", "changepoint", "source_group", "source_file",
        "date", "time", "datenum",
    }

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None

    def _resolve_feature_columns(self, train_df, feature_cols=None):
        if feature_cols is not None:
            return list(feature_cols)

        numeric_cols = train_df.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
        return [
            col for col in numeric_cols
            if col.lower() not in self.EXCLUDE_KEYWORDS
        ]

    def normalize(self, train_df, val_df, test_df, feature_cols=None):
        print("\n[Preprocessor] Normalizasyon işlemi başlatılıyor...")

        features_to_scale = self._resolve_feature_columns(train_df, feature_cols)
        print(f"[Preprocessor] Scaler fit ediliyor... (Özellik sayısı: {len(features_to_scale)})")
        self.scaler.fit(train_df[features_to_scale])

        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()

        train_scaled[features_to_scale] = self.scaler.transform(train_df[features_to_scale])
        val_scaled[features_to_scale] = self.scaler.transform(val_df[features_to_scale])
        test_scaled[features_to_scale] = self.scaler.transform(test_df[features_to_scale])

        print("[Preprocessor] Normalizasyon tamamlandı. [Data Leakage Önlemi: Aktif]")
        return train_scaled, val_scaled, test_scaled, features_to_scale

    def apply_pca(self, train_scaled, val_scaled, test_scaled, features_to_scale):
        print("\n[Preprocessor] PCA (Boyut İndirgeme) işlemi başlatılıyor...")

        self.pca = PCA(n_components=1)
        self.pca.fit(train_scaled[features_to_scale])

        train_pc1 = self.pca.transform(train_scaled[features_to_scale]).flatten()
        val_pc1 = self.pca.transform(val_scaled[features_to_scale]).flatten()
        test_pc1 = self.pca.transform(test_scaled[features_to_scale]).flatten()

        explained = self.pca.explained_variance_ratio_[0] * 100
        print(f"[Preprocessor] PCA tamamlandı. Explained Variance: %{explained:.2f}")
        return train_pc1, val_pc1, test_pc1
