import pandas as pd
import os

class DataLoader:
    def __init__(self, config):
        
        self.config = config
        self.raw_path = config['paths']['raw_data']
        self.datasets = config['project']['datasets']  # ["WADI", "BATADAL"]

    def load_data(self, dataset_name):
        
        if dataset_name not in self.datasets:
            raise ValueError(f"[Hata] {dataset_name} config'de tanımlı değil. İzin verilenler: {self.datasets}")
            
        file_path = os.path.join(self.raw_path, f"{dataset_name.lower()}.csv")
        print(f"[DataLoader] {dataset_name} veri seti aranıyor: {file_path}")
        
        try:
            
            df = pd.read_csv(file_path)
            print(f"[DataLoader] Başarılı: {dataset_name} yüklendi. Boyut: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"[Uyarı] {file_path} bulunamadı. Test edebilmek için data/raw klasörüne örnek bir {dataset_name.lower()}.csv eklemelisiniz.")
            return None
    
    def split_data(self, df):
        
        if df is None:
            return None, None, None

        
        train_ratio = self.config['data_split']['train']
        val_ratio = self.config['data_split']['validation']

        total_len = len(df)
        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)

        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        print(f"[DataLoader] Veri Bölündü -> Train: {len(train_df)} satır, Validation: {len(val_df)} satır, Test: {len(test_df)} satır")
        
        return train_df, val_df, test_df