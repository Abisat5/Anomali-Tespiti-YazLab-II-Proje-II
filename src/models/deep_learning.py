import tensorflow as tf
from tensorflow.keras import layers, models

class CNN1DAnomalyDetector:
    def __init__(self, config, seq_len, n_features):
        """
        Anlık ve yerel değişimleri yakalamada başarılı olan 1D-CNN Autoencoder Modeli.
        """
        self.config = config
        self.seq_len = seq_len
        self.n_features = n_features
        self.learning_rate = config['deep_learning']['learning_rate']
        self.model = self._build_model()

    def _build_model(self):
        """1D-CNN Autoencoder mimarisini oluşturur."""
        print("[CNN-1D] Model mimarisi inşa ediliyor...")
        model = models.Sequential(name="CNN1D_Autoencoder")
        
        # Encoder (Sıkıştırma)
        model.add(layers.Input(shape=(self.seq_len, self.n_features)))
        model.add(layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.MaxPooling1D(pool_size=2, padding="same"))
        model.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.MaxPooling1D(pool_size=2, padding="same"))
        
        # Decoder (Yeniden Oluşturma)
        model.add(layers.UpSampling1D(size=2))
        model.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.UpSampling1D(size=2))
        model.add(layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
        
        # Çıktı Katmanı (Orijinal boyuta dönüş)
        model.add(layers.Conv1D(filters=self.n_features, kernel_size=3, padding="same", activation="linear"))
        
        # Optimizasyon ve Derleme
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model

    def summary(self):
        self.model.summary()