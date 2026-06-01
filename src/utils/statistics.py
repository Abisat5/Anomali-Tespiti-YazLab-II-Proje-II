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
    def run_wilcoxon_test(self, model1_f1_scores, model2_f1_scores):
        """
        [Commit 32] İki modelin başarı farklarının istatistiksel olarak anlamlı 
        olup olmadığını PDF'te istenen Wilcoxon testi ile ölçer.
        """
        from scipy.stats import wilcoxon
        print("\n[StatAnalyzer] Wilcoxon İstatistiksel Anlamlılık Testi Uygulanıyor...")
        
        if len(model1_f1_scores) < 5 or len(model2_f1_scores) < 5:
            print("[Uyarı] Wilcoxon testi için en az 5 (seed) örneklem olmalıdır.")
            return None, None
            
        try:
            # Wilcoxon signed-rank testi
            stat, p_value = wilcoxon(model1_f1_scores, model2_f1_scores)
            significance = "Anlamlı Fark VAR (p < 0.05)" if p_value < 0.05 else "Anlamlı Fark YOK (p >= 0.05)"
            print(f"Wilcoxon Stat: {stat:.4f}, p-value: {p_value:.4f} -> {significance}")
            return p_value, significance
        except Exception as e:
            print(f"[Hata] Wilcoxon testi uygulanamadı (Değerler tamamen aynı olabilir): {e}")
            return None, None