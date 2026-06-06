import numpy as np
from scipy.stats import norm


class SAXConverter:
    def __init__(self, config):
        self.alphabet_size = config["automata"]["base_params"]["alphabet_size"]
        self.breakpoints = None
        self.mean = 0.0
        self.std = 1.0

    def _normalize(self, pc1_data):
        if self.std == 0:
            return np.zeros_like(pc1_data, dtype=float)
        return (np.asarray(pc1_data, dtype=float) - self.mean) / self.std

    def fit_transform(self, train_pc1):
        print(
            f"\n[SAXConverter] Train verisinden SAX söklüğü çıkarılıyor... "
            f"(Alfabe Boyutu: {self.alphabet_size})"
        )

        train_array = np.asarray(train_pc1, dtype=float)
        self.mean = float(np.mean(train_array))
        self.std = float(np.std(train_array))
        if self.std == 0:
            self.std = 1.0

        percentiles = np.linspace(
            1 / self.alphabet_size,
            1 - 1 / self.alphabet_size,
            self.alphabet_size - 1,
        )
        self.breakpoints = norm.ppf(percentiles)

        print(f"[SAXConverter] Train mean={self.mean:.4f}, std={self.std:.4f}")
        print(f"[SAXConverter] SAX Kesme Noktaları: {self.breakpoints}")

        return self.transform(train_pc1)

    def transform(self, pc1_data):
        normalized = self._normalize(pc1_data)
        indices = np.digitize(normalized, self.breakpoints)
        alphabet = [chr(97 + i) for i in range(self.alphabet_size)]
        sax_symbols = np.array([alphabet[idx] for idx in indices])

        print(
            f"[SAXConverter] SAX dönüşümü tamamlandı. "
            f"Örnek sekans: {''.join(sax_symbols[:10])}..."
        )
        return sax_symbols
