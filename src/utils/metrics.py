import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MetricCalculator:
    def __init__(self, logger):
        """
        Derin öğrenme modellerinin (1D-CNN, LSTM) performans metriklerini hesaplar
        ve logger aracılığıyla otomatik olarak JSON formatında kaydeder.
        """
        self.logger = logger

    def evaluate_and_log(self, model, x_test, y_test, threshold, model_name):
        """
        Model tahminlerini yapar, eşik değerine göre anomalileri bulur,
        metrikleri (Accuracy, Precision, Recall, F1-Score) hesaplar ve loglar.
        """
        print(f"\n[{model_name}] Test Verisi Üzerinde Değerlendirme Başlıyor...")

        # Modeli kullanarak yeniden oluşturma (reconstruction) yap
        predictions = model.predict(x_test)
        
        # Ortalama Kare Hata (MSE) hesapla
        mse = np.mean(np.power(x_test - predictions, 2), axis=1)
        
        # MSE, belirlenen eşik değerinden (threshold) büyükse anomali (1) say
        y_pred = (mse > threshold).astype(int)

        # PDF'te zorunlu tutulan metrikleri hesapla
        metrics = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1_Score": float(f1_score(y_test, y_pred, zero_division=0))
        }

        print(f"[{model_name}] Hesaplanan Metrikler: {metrics}")
        
        # Metrikleri logger aracılığıyla JSON dosyasına kaydet
        self.logger.log_metrics(model_name, metrics)
        
        return metrics, y_pred