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
        
        # Sadece Train verisi üzerinden öğren
        # Hedef değişkenin (label/anomali durumu) normalizasyona girmemesi için ayırıyoruz
        # SCADA verilerinde genelde son kolon veya belirli bir kolon label olur.
        # Bu aşamada tüm sayısal kolonları alıyoruz (ileride label kolonunu dışarıda bırakacağız)
        numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
        
        # Güvenlik önlemi: 'Label', 'label', 'Attack' gibi olası anomali kolonlarını normalizasyondan çıkar
        exclude_cols = [col for col in numeric_cols if col.lower() in ['label', 'attack', 'is_anomaly', 'timestamp', 'datetime']]
        features_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"[Preprocessor] Scaler fit ediliyor... (Kullanılan özellik sayısı: {len(features_to_scale)})")
        self.scaler.fit(train_df[features_to_scale])
        
        # Öğrenilen parametrelerle tüm setleri dönüştür
        # Uyarı vermemesi için kopyaları üzerinde çalışıyoruz
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()
        
        train_scaled[features_to_scale] = self.scaler.transform(train_df[features_to_scale])
        val_scaled[features_to_scale] = self.scaler.transform(val_df[features_to_scale])
        test_scaled[features_to_scale] = self.scaler.transform(test_df[features_to_scale])
        
        print("[Preprocessor] Normalizasyon (Z-Score) başarıyla tamamlandı. [Data Leakage Önlemi: Aktif]")
        
        return train_scaled, val_scaled, test_scaled, features_to_scale