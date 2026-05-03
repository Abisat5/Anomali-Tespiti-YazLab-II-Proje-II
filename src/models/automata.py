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
    
    def extract_patterns(self, sax_symbols, window_size):
        """
        Sliding Window (Kayan Pencere) mantığı ile SAX sembol dizisinden
        belirli uzunlukta alt örüntüler (kelimeler) çıkarır.
        """
        print(f"[Automata] Sliding Window (Boyut: {window_size}) ile örüntüler çıkarılıyor...")
        
        patterns = []
        # Sembol dizisi üzerinde pencereyi 1 birim kaydırarak örüntüleri oluştur
        for i in range(len(sax_symbols) - window_size + 1):
            pattern = "".join(sax_symbols[i:i + window_size])
            patterns.append(pattern)
            
        print(f"[Automata] Toplam {len(patterns)} adet örüntü dizisi çıkarıldı.")
        return patterns
    def build_transition_model(self, patterns):
        print("[Automata] Olasılıksal durum geçiş (Transition) modeli inşa ediliyor...")
        transition_counts = {}
        for i in range(len(patterns) - 1):
            current_state = patterns[i]
            next_state = patterns[i + 1]
            if current_state not in transition_counts:
                transition_counts[current_state] = {}
            if next_state not in transition_counts[current_state]:
                transition_counts[current_state][next_state] = 0
                
            transition_counts[current_state][next_state] += 1
        self.transition_probabilities = {}
        
        for current_state, transitions in transition_counts.items():
            total_transitions = sum(transitions.values())
            self.transition_probabilities[current_state] = {
                next_state: float(count / total_transitions)
                for next_state, count in transitions.items()
            }
        print(f"[Automata] Transition Modeli Tamamlandı. Toplam benzersiz durum (State) sayısı: {len(self.transition_probabilities)}")
        return self.transition_probabilities