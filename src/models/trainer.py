class ModelTrainer:
    def __init__(self, config, compiled_model):
        self.config = config
        self.model = compiled_model.model if hasattr(compiled_model, 'model') else compiled_model
        self.epochs = self.config['deep_learning']['epochs']
        self.batch_size = self.config['deep_learning']['batch_size']
        
    def setup_training(self):
        print(f"\n[Trainer] {self.model.name} için eğitim hazırlığı tamamlandı.")
        print(f"[Trainer] Hedef Epoch: {self.epochs} | Batch Size: {self.batch_size}")