import numpy as np

class ProbabilisticAutomata:
    def __init__(self, config, sax_converter):
        self.config = config
        self.window_sizes = config['automata']['variation_params']['window_sizes']
        self.sax_converter = sax_converter

    def apply_paa(self, data, segment_size):
        print(f"[Automata] PAA (Sıkıştırma) uygulanıyor... (Segment Boyutu: {segment_size})")
        
        remainder = len(data) % segment_size
        if remainder != 0:
            data_trimmed = data[:-remainder]
        else:
            data_trimmed = data
            
        paa_data = data_trimmed.reshape(-1, segment_size).mean(axis=1)
        print(f"[Automata] PAA Tamamlandı. (Orijinal: {len(data)}, Sıkıştırılmış: {len(paa_data)})")
        return paa_data

    def apply_sax(self, paa_data):
        print("[Automata] PAA verisi SAX sembollerine dönüştürülüyor...")
        return self.sax_converter.transform(paa_data)