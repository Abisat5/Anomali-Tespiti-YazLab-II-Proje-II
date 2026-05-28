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
    
    def evaluate_confidence(self, path_probability, threshold=0.05):
        """
        Hesaplanan Path Probability değerine göre otomata kararının 
        güven skorunu (Confidence Score) ve nihai kararını belirler.
        """
        # Eşik değerinin (threshold) altındaysa bu beklenmeyen bir durumdur (Anomali)
        if path_probability < threshold:
            decision = "ANOMALY"
            confidence_level = "Low"
        else:
            decision = "NORMAL"
            confidence_level = "High"
            
        return decision, f"{path_probability:.4f} ({confidence_level})"
    
    def calculate_path_probability(self, sequence_patterns):
        """
        Bir örüntü dizisinin toplam olasılığını (Path Probability), ardışık 
        geçiş olasılıklarının (transition probabilities) çarpımı ile hesaplar.
        """
        if not sequence_patterns or len(sequence_patterns) < 2:
            return 1.0 # Tek durum varsa geçiş yoktur

        path_prob = 1.0
        transitions_made = []

        for i in range(len(sequence_patterns) - 1):
            current_state = sequence_patterns[i]
            next_state = sequence_patterns[i+1]

            # Eğer eğitimde böyle bir geçiş hiç görülmediyse olasılık 0 (veya çok küçük bir epsilon) olur
            try:
                prob = self.transition_probabilities[current_state][next_state]
            except KeyError:
                prob = 0.001 # Smoothing (Hiç görülmemiş geçiş)
                
            path_prob *= prob
            transitions_made.append(f"{current_state} -> {next_state}: {prob:.2f}")

        return path_prob, transitions_made
    def generate_explanation_json(self, time_step, current_state, incoming_pattern, is_unseen, mapped_state, path_prob, decision):
        """
        Proje gereksinimlerinde zorunlu tutulan JSON formatında açıklanabilirlik logu üretir.
        """
        explanation = {
            "time_step": time_step,
            "state": current_state,
            "pattern": incoming_pattern,
            "status": "unseen" if is_unseen else "known",
            "mapped_to": mapped_state if is_unseen else current_state,
            "probability": round(path_prob, 4),
            "decision": decision.lower()
        }
        
        # Bu log daha sonra logger.py aracılığıyla config'de belirtilen yere kaydedilecek
        return explanation