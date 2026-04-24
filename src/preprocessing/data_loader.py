import pandas as pd
import os

class DataLoader:
    def __init__(self, config):
        """
        config.json üzerinden dosya yollarını ve veri seti isimlerini alır.
        """
        self.config = config
        self.raw_path = config['paths']['raw_data']
        self.datasets = config['project']['datasets']  # ["WADI", "BATADAL"]

    def load_data(self, dataset_name):
        """
        Belirtilen veri setini raw klasöründen pandas DataFrame olarak okur.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"[Hata] {dataset_name} config'de tanımlı değil. İzin verilenler: {self.datasets}")
            
        file_path = os.path.join(self.raw_path, f"{dataset_name.lower()}.csv")
        print(f"[DataLoader] {dataset_name} veri seti aranıyor: {file_path}")
        
        try:
            # SCADA verileri genellikle virgül ayrılıdır, dosya boyutları büyük olabilir
            df = pd.read_csv(file_path)
            print(f"[DataLoader] Başarılı: {dataset_name} yüklendi. Boyut: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"[Uyarı] {file_path} bulunamadı. Test edebilmek için data/raw klasörüne örnek bir {dataset_name.lower()}.csv eklemelisiniz.")
            return None