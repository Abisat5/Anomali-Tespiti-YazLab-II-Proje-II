from src.utils.seed_manager import set_deterministic_seed
class Pipeline:
    def __init__(self, config):
        
        self.config = config
        self.group_id = self.config['project']['group_id']
        self.random_seeds = self.config['experiment_settings']['random_seeds']
        
    def prepare_data(self):
        
        print("[Pipeline] Veri ön işleme adımı başlatıldı...")
        pass
        
    def run_deep_learning_models(self):
        
        print("[Pipeline] Derin öğrenme modelleri (1D-CNN, LSTM) çalıştırılıyor...")
        pass
        
    def run_automata_model(self):
        
        print("[Pipeline] Olasılıksal Otomata modeli inşa ediliyor...")
        pass
        
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

        print("\n--- Tüm Seed Deneyleri (Pipeline Akışı) Tamamlandı ---\n")