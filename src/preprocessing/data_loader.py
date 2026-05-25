import pandas as pd
import os
import glob

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.raw_path = config['paths']['raw_data']
        self.datasets = config['project']['datasets']
        self.skab_folders = config['paths'].get('skab_subfolders', ['valve1', 'valve2'])

    def load_data(self, dataset_name):
        if dataset_name not in self.datasets:
            raise ValueError(f"[Hata] {dataset_name} config'de tanımlı değil.")
            
        print(f"\n[DataLoader] {dataset_name} veri seti yükleniyor...")
        if dataset_name == "SKAB":
            return self._load_skab()
        elif dataset_name == "BATADAL":
            return self._load_batadal()

    def _load_skab(self):
        all_data = []
        skab_base_path = os.path.join(self.raw_path, "SKAB")

        for folder in self.skab_folders:
            folder_path = os.path.join(skab_base_path, folder)
            if not os.path.exists(folder_path):
                print(f"[Uyarı] SKAB klasörü bulunamadı: {folder_path}. Lütfen data/raw/SKAB/{folder} dizinini oluşturun.")
                continue

            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            for file_path in csv_files:
                df = pd.read_csv(file_path, sep=';') 
                
                df['source_group'] = folder
                df['source_file'] = os.path.basename(file_path)
                all_data.append(df)

        if not all_data:
            return None

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"[DataLoader] SKAB başarıyla birleştirildi (Concat). Toplam Boyut: {combined_df.shape}")
        return combined_df

    def _load_batadal(self):
        file_path = os.path.join(self.raw_path, "batadal.csv")
        try:
            df = pd.read_csv(file_path)
            print(f"[DataLoader] BATADAL yüklendi. Boyut: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"[Uyarı] {file_path} bulunamadı.")
            return None

    def split_data(self, df, dataset_name):
        """
        Veri setine özel kurallarla (BATADAL: Kronolojik, SKAB: GroupKFold) veriyi böler.
        """
        if df is None: return None, None, None
        
        print(f"\n[DataLoader] {dataset_name} için veri bölme işlemi (Split) başlatılıyor...")

        if dataset_name == "BATADAL":
            # BATADAL için zorunlu kural: Zaman sırası bozulmadan %60 - %20 - %20
            train_ratio = self.config['data_split']['train']
            val_ratio = self.config['data_split']['validation']
            
            total_len = len(df)
            train_end = int(total_len * train_ratio)
            val_end = train_end + int(total_len * val_ratio)
            
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
            
            print(f"[DataLoader - BATADAL] Kronolojik Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return train_df, val_df, test_df

        elif dataset_name == "SKAB":
            # SKAB için zorunlu kural: Aynı .csv dosyası hem train hem testte olamaz! (GroupKFold mantığı)
            # Şimdilik ana pipeline için dosyaların %60'ını Train, %20'sini Val, %20'sini Test yapacağız.
            # (Cross-Validation döngüsü test aşamasında ayrıca çağrılacaktır).
            unique_files = df['source_file'].unique()
            
            train_ratio = self.config['data_split']['train']
            val_ratio = self.config['data_split']['validation']
            
            n_train = int(len(unique_files) * train_ratio)
            n_val = int(len(unique_files) * val_ratio)
            
            train_files = unique_files[:n_train]
            val_files = unique_files[n_train:n_train + n_val]
            test_files = unique_files[n_train + n_val:]
            
            train_df = df[df['source_file'].isin(train_files)].copy()
            val_df = df[df['source_file'].isin(val_files)].copy()
            test_df = df[df['source_file'].isin(test_files)].copy()
            
            print(f"[DataLoader - SKAB] Dosya Bazlı Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return train_df, val_df, test_df