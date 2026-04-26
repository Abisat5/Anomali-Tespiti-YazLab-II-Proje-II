import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        """
        Zaman serisi verileri için standartlaştırma ve boyut indirgeme sınıfı.
        """
        self.scaler = StandardScaler()
        
    def normalize(self, train_df, val_df, test_df):
        """
        Veri sızıntısını (data leakage) önlemek için scaler SADECE eğitim
        verisi üzerinde fit (öğrenme) işlemi yapar. Öğrenilen bu parametreler
        (ortalama ve varyans) ile tüm veri setleri transform edilir.
        """
        print("\n[Preprocessor] Normalizasyon işlemi başlatılıyor...")
        
        
        numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
        
        
        exclude_cols = [col for col in numeric_cols if col.lower() in ['label', 'attack', 'is_anomaly', 'timestamp', 'datetime']]
        features_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"[Preprocessor] Scaler fit ediliyor... (Kullanılan özellik sayısı: {len(features_to_scale)})")
        self.scaler.fit(train_df[features_to_scale])
        
        
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()
        
        train_scaled[features_to_scale] = self.scaler.transform(train_df[features_to_scale])
        val_scaled[features_to_scale] = self.scaler.transform(val_df[features_to_scale])
        test_scaled[features_to_scale] = self.scaler.transform(test_df[features_to_scale])
        
        print("[Preprocessor] Normalizasyon (Z-Score) başarıyla tamamlandı. [Data Leakage Önlemi: Aktif]")
        
        return train_scaled, val_scaled, test_scaled, features_to_scale
    
    def apply_pca(self, train_scaled, val_scaled, test_scaled, features_to_scale):
        """
        Otomata modeli tek boyutlu veriyle çalıştığı için çok değişkenli sensör
        verilerini PCA ile tek boyuta (PC1) indirger.
        Data Leakage önlemi: PCA yalnızca Train verisi üzerinde fit edilir.
        """
        from sklearn.decomposition import PCA
        
        print("\n[Preprocessor] PCA (Boyut İndirgeme) işlemi başlatılıyor...")
        
        
        self.pca = PCA(n_components=1)
        
        print("[Preprocessor] PCA sadece Train verisi üzerinde fit ediliyor...")
        self.pca.fit(train_scaled[features_to_scale])
        
        
        train_pc1 = self.pca.transform(train_scaled[features_to_scale]).flatten()
        val_pc1 = self.pca.transform(val_scaled[features_to_scale]).flatten()
        test_pc1 = self.pca.transform(test_scaled[features_to_scale]).flatten()
        
        print(f"[Preprocessor] PCA tamamlandı. Açıklanan Varyans (Explained Variance) Oranı: %{self.pca.explained_variance_ratio_[0]*100:.2f}")
        print("[Preprocessor] Tüm veri setleri başarılı bir şekilde PC1 (Tek Boyut) formatına dönüştürüldü.")
        
        return train_pc1, val_pc1, test_pc1