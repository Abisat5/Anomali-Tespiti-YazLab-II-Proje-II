import numpy as np

class StatisticalAnalyzer:
    def __init__(self):
        """Deney tekrarlarından (seeds) elde edilen metriklerin istatistiksel analizini yapar."""
        pass

    def calculate_mean_and_std(self, metrics_list):
        """
        [Commit 31] Farklı seed değerlerinden elde edilen metrik listesini alır, 
        ortalama (mean) ve standart sapma (std) değerlerini hesaplar.
        """
        print("\n[StatAnalyzer] Ortalama ve Standart Sapma hesaplanıyor...")
        aggregated = {}
        
        if not metrics_list:
            return aggregated
            
        # metrics_list örneğin: [{'Accuracy': 0.9, 'F1': 0.8}, {'Accuracy': 0.92, 'F1': 0.85}]
        keys = metrics_list[0].keys()
        
        for key in keys:
            values = [m[key] for m in metrics_list]
            mean_val = np.mean(values)
            std_val = np.std(values)
            # Raporlama formatı: Ortalama ± Standart Sapma
            aggregated[key] = f"{mean_val:.4f} ± {std_val:.4f}"
            
        return aggregated