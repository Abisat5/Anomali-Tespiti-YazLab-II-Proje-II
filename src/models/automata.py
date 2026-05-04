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
    def calculate_levenshtein_distance(self, s1, s2):
        """
        İki string (örüntü) arasındaki minimum düzenleme mesafesini 
        (Ekleme, Silme, Değiştirme) dinamik programlama ile hesaplar.
        """
        if len(s1) < len(s2):
            return self.calculate_levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def handle_unseen_pattern(self, unseen_pattern):
        """
        Daha önce görülmemiş bir örüntü geldiğinde, eğitim setinden öğrenilen
        en yakın (Levenshtein mesafesi en küçük) örüntüyü bulur.
        """
        closest_pattern = None
        min_distance = float('inf')

        # Eğitimde öğrenilen tüm durumları gez
        for known_pattern in self.transition_probabilities.keys():
            dist = self.calculate_levenshtein_distance(unseen_pattern, known_pattern)
            
            if dist < min_distance:
                min_distance = dist
                closest_pattern = known_pattern
                
            # Eğer birebir eşleşme (mesafe=0) bulunursa aramayı kes
            if min_distance == 0:
                break
                
        print(f"[Automata - Unseen] '{unseen_pattern}' -> '{closest_pattern}' ile eşleştirildi (Mesafe: {min_distance})")
        return closest_pattern, min_distance