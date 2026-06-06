import os
from src.utils.seed_manager import set_deterministic_seed
from src.preprocessing.noise_injector import NoiseInjector
from src.utils.visualizer import Visualizer
from src.utils.statistics import StatisticalAnalyzer
from src.utils.analyzer import ParameterAnalyzer

class Pipeline:
    def __init__(self, config):
        """Pipeline sınıfı, tüm sistemin modüler olarak yönetildiği yerdir."""
        self.config = config
        self.group_id = self.config['project']['group_id']
        self.random_seeds = self.config['experiment_settings']['random_seeds']
        
        # Yeni Eklenen Modüllerin Başlatılması
        self.visualizer = Visualizer(config)
        self.stat_analyzer = StatisticalAnalyzer()
        
        # Deney sonuçlarını seed bazlı tutacağımız listeler
        self.lstm_f1_scores = []
        self.cnn_f1_scores = []
        
    def prepare_data(self):
        print("[Pipeline] Veri ön işleme adımı (DataLoader, Normalizasyon, PCA, SAX) çalıştırılıyor...")
        pass
        
    def run_deep_learning_models(self):
        print("[Pipeline] Derin öğrenme modelleri (1D-CNN, LSTM) eğitiliyor...")
        pass
        
    def run_automata_model(self):
        print("[Pipeline] Olasılıksal Otomata modeli inşa ediliyor...")
        pass
        
    def run_experimental_scenarios(self):
        """Zorunlu 3 senaryoyu, görselleri ve analizleri tetikler."""
        print("\n[Pipeline] Deneysel Senaryolar Başlatılıyor...")
        noise_injector = NoiseInjector(self.config)
        
        print("\n>>> Senaryo 1: Orijinal Veri Performansı ve Görseller <<<")
        # Not: Gerçek model tahminleri (y_pred, y_true) buralara bağlanacak
        # self.visualizer.plot_confusion_matrix(y_true, y_pred, "Otomata")
        # self.visualizer.plot_roc_curve(y_true, y_scores, "Otomata")
        
        print("\n>>> Senaryo 2: Gürültü (Gaussian Noise) Eklenmiş Veri Testi <<<")
        # noisy_data = noise_injector.inject_noise(test_data)
        
        print("\n>>> Senaryo 3: Unseen Pattern, Counterfactual ve Heatmap <<<")
        # self.visualizer.plot_transition_heatmap(transition_probs)
        # self.visualizer.plot_state_diagram(transition_probs)
        
    def run_explainability_module(self):
        print("[Pipeline] Açıklanabilirlik ve JSON loglama modülü çalıştırılıyor...")
        pass
        
    def run(self):
        print(f"\n{'='*50}")
        print(f"--- {self.group_id}. Grup Pipeline Akışı Başlıyor ---")
        print(f"{'='*50}")

        for seed in self.random_seeds:
            print(f"\n>>> [Deney Döngüsü] Random Seed: {seed} ile çalıştırılıyor <<<")
            set_deterministic_seed(seed)
            
            self.prepare_data()
            self.run_deep_learning_models()
            self.run_automata_model()
            self.run_experimental_scenarios()
            self.run_explainability_module()

        print("\n--- Tüm Seed Deneyleri Tamamlandı. İstatistiksel Analizlere Geçiliyor ---")
        
        # 5 Seed bittikten sonra Wilcoxon Testi tetiklenir
        # self.stat_analyzer.run_wilcoxon_test(self.lstm_f1_scores, self.cnn_f1_scores)
        
        print("\n--- Pipeline Akışı Başarıyla Tamamlandı ---\n")