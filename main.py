import json
import os
from src.pipeline import Pipeline

def load_config(config_path="configs/config.json"):
    """Merkezi konfigürasyon dosyasını okur."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("YazLab-II Proje 1 Başlatılıyor...")
    
    # 1. Merkezi konfigürasyonu yükle
    config = load_config()
    print(f"Grup {config['project']['group_id']} - Veri Setleri: {config['project']['datasets']} yükleniyor.")
    
    # 2. Pipeline'ı başlat
    anomali_pipeline = Pipeline(config)
    
    # 3. Akışı çalıştır (Şimdilik sadece iskelet)
    anomali_pipeline.run()

if __name__ == "__main__":
    main()