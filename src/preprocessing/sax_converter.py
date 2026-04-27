import numpy as np
from scipy.stats import norm

class SAXConverter:
    def __init__(self, config):
        """
        Otomata tabanlı model için sürekli zaman serilerini sembolik verilere
        (harflere) dönüştüren SAX (Symbolic Aggregate approXimation) sınıfı.
        """
        self.alphabet_size = config['automata']['base_params']['alphabet_size']
        self.breakpoints = None

    def fit_transform(self, train_pc1):
        """
        [Data Leakage Önlemi] Sınır değerlerini (breakpoints) SADECE Train verisinin 
        istatistiksel dağılımı (veya standart normal dağılım) üzerinden belirler.
        """
        print(f"\n[SAXConverter] Train verisinden SAX sözlüğü çıkarılıyor... (Alfabe Boyutu: {self.alphabet_size})")
        
        # Standart normal dağılım eğrisi altında eşit alanlar oluşturacak sınırları bul
        # Örn: alphabet_size = 3 ise, alanı 3'e bölen 2 adet kesme noktası (breakpoint) bulur
        percentiles = np.linspace(1 / self.alphabet_size, 1 - 1 / self.alphabet_size, self.alphabet_size - 1)
        self.breakpoints = norm.ppf(percentiles)
        
        print(f"[SAXConverter] SAX Kesme Noktaları (Breakpoints) öğrenildi: {self.breakpoints}")
        
        return self.transform(train_pc1)

    def transform(self, pc1_data):
        """
        Öğrenilmiş sınır değerleri ile Train, Val veya Test verisini sembollere çevirir.
        """
        # Veriyi sınır noktalarına göre indekslere (0, 1, 2...) ayır
        indices = np.digitize(pc1_data, self.breakpoints)
        
        # İndeksleri ASCII karakterlerine çevir (0->'a', 1->'b', 2->'c' ...)
        alphabet = [chr(97 + i) for i in range(self.alphabet_size)]
        
        # NumPy string dizisine dönüştür
        sax_symbols = np.array([alphabet[idx] for idx in indices])
        
        print(f"[SAXConverter] SAX dönüşümü tamamlandı. Örnek sekans: {''.join(sax_symbols[:10])}...")
        
        return sax_symbols