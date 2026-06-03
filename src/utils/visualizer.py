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