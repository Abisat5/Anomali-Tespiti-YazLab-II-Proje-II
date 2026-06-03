import pandas as pd

class ParameterAnalyzer:
    def __init__(self, config, base_automata):
        """Otomata parametre değişimlerinin etkisini test eden analiz modülü."""
        self.config = config
        self.automata = base_automata
        self.window_sizes = config['automata']['variation_params']['window_sizes']
        self.alphabet_sizes = config['automata']['variation_params']['alphabet_sizes']

    def analyze_window_size(self, sax_symbols):
        """
        [Commit 33] Window Size (3, 4, 5, 6) etkilerini analiz eder.
        """
        print("\n[ParameterAnalyzer] Window Size (Pencere Boyutu) duyarlılık analizi başlatıldı...")
        results = []
        
        for w_size in self.window_sizes:
            # Modelin pattern çıkarma işlemini dinamik window_size'a göre tekrarla
            patterns = self.automata.extract_patterns(sax_symbols, window_size=w_size)
            transitions = self.automata.build_transition_model(patterns)
            
            state_count = len(transitions)
            # Geçiş yoğunluğu = Toplam mevcut geçiş / Olası tüm geçişler (basit bir yoğunluk oranı)
            total_transitions = sum([len(v) for v in transitions.values()])
            transition_density = total_transitions / (state_count + 1e-5) # Sıfıra bölünmeyi engelle
            
            results.append({
                "Parameter": "Window_Size", 
                "Value": w_size, 
                "State_Count": state_count, 
                "Transition_Density": round(transition_density, 3)
            })
            print(f"  -> Window {w_size}: State Sayısı={state_count}, Yoğunluk={transition_density:.3f}")
            
        return results
    def analyze_alphabet_size(self, original_pc1_data, sax_converter_class):
        """
        [Commit 34] Alphabet Size (3, 4, 5, 6) etkilerini analiz eder.
        Alfabe değiştiğinde SAX sözlüğünün baştan çıkarılması gerekir.
        """
        print("\n[ParameterAnalyzer] Alphabet Size (Alfabe Boyutu) duyarlılık analizi başlatıldı...")
        results = []
        
        # Orijinal config'i bozmamak için kopyala
        temp_config = self.config.copy()
        
        for a_size in self.alphabet_sizes:
            temp_config['automata']['base_params']['alphabet_size'] = a_size
            
            # Yeni alfabeye göre SAX dönüştürücü oluştur ve fit et
            temp_sax = sax_converter_class(temp_config)
            new_sax_symbols = temp_sax.fit_transform(original_pc1_data)
            
            # Patternleri çıkar ve transition modelini kur (sabit window size = 4 ile)
            patterns = self.automata.extract_patterns(new_sax_symbols, window_size=4)
            transitions = self.automata.build_transition_model(patterns)
            
            state_count = len(transitions)
            total_transitions = sum([len(v) for v in transitions.values()])
            transition_density = total_transitions / (state_count + 1e-5)
            
            results.append({
                "Parameter": "Alphabet_Size", 
                "Value": a_size, 
                "State_Count": state_count, 
                "Transition_Density": round(transition_density, 3)
            })
            print(f"  -> Alphabet {a_size}: State Sayısı={state_count}, Yoğunluk={transition_density:.3f}")
            
        return results