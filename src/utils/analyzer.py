import copy
import os

import pandas as pd

from src.models.automata import ProbabilisticAutomata
from src.preprocessing.sax_converter import SAXConverter


class ParameterAnalyzer:
    def __init__(self, config, base_automata):
        self.config = config
        self.automata = base_automata
        self.window_sizes = config["automata"]["variation_params"]["window_sizes"]
        self.alphabet_sizes = config["automata"]["variation_params"]["alphabet_sizes"]
        self.base_window = config["automata"]["base_params"]["window_size"]

    def _evaluate_automata_performance(self, train_pc1, test_pc1, test_labels, window_size, sax_converter):
        automata = ProbabilisticAutomata(self.config, sax_converter)
        automata.fit(train_pc1, window_size=window_size)
        y_true, y_pred, _, _, _ = automata.predict_from_pc1(test_pc1, test_labels)
        return automata.evaluate_metrics(y_true, y_pred)

    def analyze_window_size(self, train_pc1, test_pc1, test_labels, sax_converter):
        print("\n[ParameterAnalyzer] Window Size duyarlılık analizi başlatıldı...")
        results = []

        for w_size in self.window_sizes:
            patterns = self.automata.extract_patterns(
                sax_converter.transform(self.automata.apply_paa(train_pc1)),
                window_size=w_size,
            )
            transitions = self.automata.build_transition_model(patterns)

            state_count = len(transitions)
            total_transitions = sum(len(v) for v in transitions.values())
            transition_density = total_transitions / (state_count + 1e-5)

            performance = self._evaluate_automata_performance(
                train_pc1, test_pc1, test_labels, w_size, sax_converter
            )

            results.append({
                "Parameter": "Window_Size",
                "Value": w_size,
                "State_Count": state_count,
                "Transition_Density": round(transition_density, 3),
                "Accuracy": round(performance["Accuracy"], 4),
                "F1_Score": round(performance["F1_Score"], 4),
            })
            print(
                f"  -> Window {w_size}: State={state_count}, "
                f"Yoğunluk={transition_density:.3f}, F1={performance['F1_Score']:.4f}"
            )

        return results

    def analyze_alphabet_size(self, train_pc1, test_pc1, test_labels, sax_converter_class):
        print("\n[ParameterAnalyzer] Alphabet Size duyarlılık analizi başlatıldı...")
        results = []

        for a_size in self.alphabet_sizes:
            temp_config = copy.deepcopy(self.config)
            temp_config["automata"]["base_params"]["alphabet_size"] = a_size

            temp_sax = sax_converter_class(temp_config)
            temp_sax.fit_transform(train_pc1)

            patterns = self.automata.extract_patterns(
                temp_sax.transform(self.automata.apply_paa(train_pc1)),
                window_size=self.base_window,
            )
            transitions = self.automata.build_transition_model(patterns)

            state_count = len(transitions)
            total_transitions = sum(len(v) for v in transitions.values())
            transition_density = total_transitions / (state_count + 1e-5)

            performance = self._evaluate_automata_performance(
                train_pc1, test_pc1, test_labels, self.base_window, temp_sax
            )

            results.append({
                "Parameter": "Alphabet_Size",
                "Value": a_size,
                "State_Count": state_count,
                "Transition_Density": round(transition_density, 3),
                "Accuracy": round(performance["Accuracy"], 4),
                "F1_Score": round(performance["F1_Score"], 4),
            })
            print(
                f"  -> Alphabet {a_size}: State={state_count}, "
                f"Yoğunluk={transition_density:.3f}, F1={performance['F1_Score']:.4f}"
            )

        return results

    def export_analysis_to_table(self, all_results, filepath="logs/parameter_analysis.csv"):
        print(f"\n[ParameterAnalyzer] Sonuçlar dışa aktarılıyor: {filepath}")

        if not all_results:
            print("[Uyarı] Dışa aktarılacak analiz sonucu bulunamadı.")
            return None

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(filepath, index=False)
        print("[ParameterAnalyzer] Tablo kaydedildi.")
        return df_results
