import json
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, config):
        self.config = config
        self.log_dir = config['paths']['logs']
        
        # Log klasörü yoksa oluştur
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Her deney için benzersiz bir timestamp (zaman damgası)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"exp_{self.experiment_id}.json")
        
        # Ana log yapısı
        self.results = {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "metrics": {},
            "explainability": []
        }

    def log_metrics(self, model_name, metrics):
        """Model performans metriklerini (Accuracy, Precision, Recall, F1) kaydeder."""
        self.results["metrics"][model_name] = metrics
        self.save()

    def log_explainability(self, decision_data):
        """Olasılıksal otomata ve unseen JSON çıktılarını kaydeder."""
        self.results["explainability"].append(decision_data)
        self.save()

    def save(self):
        """Verileri JSON formatında diske yazar."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        print(f"[Logger] Deney sonuçları JSON formatında güncellendi: {self.log_file}")