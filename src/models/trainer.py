class ModelTrainer:
    def __init__(self, config, compiled_model):
        self.config = config
        self.model = compiled_model.model if hasattr(compiled_model, 'model') else compiled_model
        self.epochs = self.config['deep_learning']['epochs']
        self.batch_size = self.config['deep_learning']['batch_size']
        
    def setup_training(self):
        print(f"\n[Trainer] {self.model.name} için eğitim hazırlığı tamamlandı.")
        print(f"[Trainer] Hedef Epoch: {self.epochs} | Batch Size: {self.batch_size}")
        
    def train(self, X_train, X_val):
        """
        Autoencoder modelini zaman serisi sırasını bozmadan eğitir ve 
        Early Stopping (Erken Durdurma) mekanizması uygular.
        """
        from tensorflow.keras.callbacks import EarlyStopping
        
        patience = self.config['deep_learning']['early_stopping_patience']
        
        # Validation loss'u izle, gelişme yoksa eğitimi kes ve en iyi ağırlıklara dön
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        print(f"[Trainer] {self.model.name} eğitimi başlatılıyor... (Early Stopping Patience: {patience})")
        
        # Autoencoder mantığı: Girdi X, hedef de X (kendi kendini kopyalamaya çalışır)
        history = self.model.fit(
            X_train, X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            shuffle=False,  # Zaman serisinde veri sırası KESİNLİKLE bozulmaz!
            verbose=1
        )
        
        print(f"[Trainer] {self.model.name} eğitimi tamamlandı.")
        return history