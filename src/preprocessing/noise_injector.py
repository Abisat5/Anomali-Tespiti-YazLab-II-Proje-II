import numpy as np

class NoiseInjector:
    def __init__(self, config):
        self.noise_config = config['experiment_settings']['noise_injection']
        self.mean = self.noise_config['mean']
        self.std_dev = self.noise_config['std_dev']

    def inject_noise(self, data):
        if data is None:
            return None
            
        print(f"[NoiseInjector] Veriye Gaussian Noise ekleniyor (Ortalama: {self.mean}, Std Sapma: {self.std_dev})")
        
        
        noise = np.random.normal(self.mean, self.std_dev, data.shape)
        noisy_data = data + noise
        
        return noisy_data