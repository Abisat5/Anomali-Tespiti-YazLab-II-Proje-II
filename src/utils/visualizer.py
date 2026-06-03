import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

class Visualizer:
    def __init__(self, config):
        self.log_dir = config['paths']['logs']
        os.makedirs(self.log_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        from sklearn.metrics import confusion_matrix
        print(f"\n[Visualizer] {model_name} için Confusion Matrix oluşturuluyor...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'], 
                    yticklabels=['Normal', 'Anomaly'])
        
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('Gerçek Etiket')
        plt.xlabel('Tahmin Edilen Etiket')
        
        save_path = os.path.join(self.log_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[Visualizer] Grafik kaydedildi: {save_path}")
    
    def plot_roc_curve(self, y_true, y_scores, model_name):
        from sklearn.metrics import roc_curve, auc
        print(f"[Visualizer] {model_name} için ROC eğrisi oluşturuluyor...")
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
            plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
            plt.title(f'{model_name} - ROC Eğrisi')
            plt.legend(loc="lower right")
            
            save_path = os.path.join(self.log_dir, f"{model_name}_roc_curve.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"[Visualizer] Grafik kaydedildi: {save_path}")
        except Exception as e:
            print(f"[Hata] ROC eğrisi çizilemedi (Skorlar eksik veya hatalı olabilir): {e}")