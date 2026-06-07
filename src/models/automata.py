import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ProbabilisticAutomata:
    def __init__(self, config, sax_converter):
        self.config = config
        automata_cfg = config["automata"]
        base_params = automata_cfg.get("base_params", {})
        self.window_sizes = automata_cfg["variation_params"]["window_sizes"]
        self.sax_converter = sax_converter
        self.transition_probabilities = {}
        self.train_pattern_set = set()
        self.window_size = base_params.get("window_size", 4)
        self.paa_segment_size = base_params.get("paa_segment_size", 1)
        self.threshold = base_params.get("path_probability_threshold", 0.05)
        self.smoothing_alpha = base_params.get("smoothing_alpha", 1.0)
        self.unknown_transition_prob = base_params.get("unknown_transition_prob", 0.001)

    def apply_paa(self, data, segment_size=None):
        segment_size = segment_size or self.paa_segment_size
        print(f"[Automata] PAA uygulanıyor... (Segment Boyutu: {segment_size})")

        data = np.asarray(data, dtype=float)
        remainder = len(data) % segment_size
        data_trimmed = data[:-remainder] if remainder else data
        if len(data_trimmed) == 0:
            return data_trimmed

        paa_data = data_trimmed.reshape(-1, segment_size).mean(axis=1)
        print(f"[Automata] PAA Tamamlandı. (Orijinal: {len(data)}, Sıkıştırılmış: {len(paa_data)})")
        return paa_data

    def apply_sax(self, paa_data):
        print("[Automata] PAA verisi SAX sembollerine dönüştürülüyor...")
        return self.sax_converter.transform(paa_data)

    def extract_patterns(self, sax_symbols, window_size=None):
        window_size = window_size or self.window_size
        print(f"[Automata] Sliding Window (Boyut: {window_size}) ile örüntüler çıkarılıyor...")

        patterns = []
        for i in range(len(sax_symbols) - window_size + 1):
            patterns.append("".join(sax_symbols[i:i + window_size]))

        print(f"[Automata] Toplam {len(patterns)} adet örüntü çıkarıldı.")
        return patterns

    def build_transition_model(self, patterns):
        print("[Automata] Olasılıksal durum geçiş modeli inşa ediliyor...")
        transition_counts = {}

        for i in range(len(patterns) - 1):
            current_state = patterns[i]
            next_state = patterns[i + 1]
            transition_counts.setdefault(current_state, {})
            transition_counts[current_state][next_state] = (
                transition_counts[current_state].get(next_state, 0) + 1
            )

        alpha = self.smoothing_alpha
        self.transition_probabilities = {}
        for current_state, transitions in transition_counts.items():
            total_transitions = sum(transitions.values())
            num_targets = len(transitions)
            self.transition_probabilities[current_state] = {
                next_state: float((count + alpha) / (total_transitions + alpha * num_targets))
                for next_state, count in transitions.items()
            }

        print(
            f"[Automata] Transition modeli tamamlandı (smoothing={alpha}). "
            f"Benzersiz state sayısı: {len(self.transition_probabilities)}"
        )
        return self.transition_probabilities

    def get_transition_probability(self, from_state, to_state):
        try:
            return self.transition_probabilities[from_state][to_state]
        except KeyError:
            return self.unknown_transition_prob

    def fit(self, train_pc1, window_size=None):
        self.window_size = window_size or self.config["automata"]["base_params"]["window_size"]
        paa_train = self.apply_paa(train_pc1)
        sax_train = self.apply_sax(paa_train)
        train_patterns = self.extract_patterns(sax_train, self.window_size)
        self.train_pattern_set = set(train_patterns)
        self.build_transition_model(train_patterns)
        return train_patterns, sax_train

    def resolve_pattern(self, pattern, use_unseen_mapping=True):
        is_unseen = pattern not in self.train_pattern_set
        mapped_pattern = pattern
        distance = 0

        if is_unseen and use_unseen_mapping:
            mapped_pattern, distance = self.handle_unseen_pattern(pattern)

        return mapped_pattern, is_unseen, distance

    def calculate_levenshtein_distance(self, s1, s2):
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
        closest_pattern = None
        min_distance = float("inf")

        for known_pattern in self.transition_probabilities.keys():
            dist = self.calculate_levenshtein_distance(unseen_pattern, known_pattern)
            if dist < min_distance:
                min_distance = dist
                closest_pattern = known_pattern
            if min_distance == 0:
                break

        print(
            f"[Automata - Unseen] '{unseen_pattern}' -> '{closest_pattern}' "
            f"ile eşleştirildi (Mesafe: {min_distance})"
        )
        return closest_pattern, min_distance

    def evaluate_confidence(self, path_probability, threshold=None):
        threshold = threshold or self.threshold
        if path_probability < threshold:
            return "ANOMALY", path_probability
        return "NORMAL", path_probability

    def calculate_path_probability(self, sequence_patterns):
        if not sequence_patterns or len(sequence_patterns) < 2:
            return 1.0, []

        path_prob = 1.0
        transitions_made = []

        for i in range(len(sequence_patterns) - 1):
            current_state = sequence_patterns[i]
            next_state = sequence_patterns[i + 1]
            prob = self.get_transition_probability(current_state, next_state)

            path_prob *= prob
            transitions_made.append(
                {"from": current_state, "to": next_state, "probability": round(prob, 4)}
            )

        return path_prob, transitions_made

    def predict_from_pc1(self, pc1_data, labels, use_unseen_mapping=True):
        paa_data = self.apply_paa(pc1_data)
        sax_symbols = self.apply_sax(paa_data)
        patterns = self.extract_patterns(sax_symbols, self.window_size)

        if len(patterns) < 2:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                [],
                {"unseen_count": 0, "total_patterns": 0, "detection_rate": 0.0, "mapping_accuracy": 0.0},
            )

        label_offset = self.window_size - 1
        y_true = []
        y_pred = []
        anomaly_scores = []
        explanations = []
        unseen_count = 0
        mapping_scores = []
        unseen_anomaly_hits = 0

        for idx in range(1, len(patterns)):
            prev_pattern, _, _ = self.resolve_pattern(patterns[idx - 1], use_unseen_mapping)
            current_pattern, is_unseen, distance = self.resolve_pattern(
                patterns[idx], use_unseen_mapping
            )
            if is_unseen:
                unseen_count += 1
                mapping_scores.append(max(0.0, 1.0 - distance / self.window_size))

            mapped_current = current_pattern if is_unseen else patterns[idx]
            path_prob, transitions = self.calculate_path_probability(
                [prev_pattern, mapped_current]
            )
            decision, confidence = self.evaluate_confidence(path_prob)

            if is_unseen and decision == "ANOMALY":
                unseen_anomaly_hits += 1

            label_index = min(idx + label_offset, len(labels) - 1)
            y_true.append(int(labels[label_index]))
            y_pred.append(1 if decision == "ANOMALY" else 0)
            anomaly_scores.append(1.0 - path_prob)

            explanations.append(
                self.generate_explanation_json(
                    time_step=idx,
                    current_state=prev_pattern,
                    incoming_pattern=patterns[idx],
                    is_unseen=is_unseen,
                    mapped_state=mapped_current,
                    path_prob=path_prob,
                    decision=decision,
                    transitions=transitions,
                    distance=distance if is_unseen else 0,
                    confidence=confidence,
                )
            )

        evaluated = max(len(patterns) - 1, 1)
        unseen_stats = {
            "unseen_count": unseen_count,
            "total_patterns": len(patterns),
            "unseen_ratio": round(unseen_count / max(len(patterns), 1), 4),
            "detection_rate": round(unseen_count / evaluated, 4),
            "mapping_accuracy": round(float(np.mean(mapping_scores)), 4) if mapping_scores else 1.0,
            "unseen_anomaly_detection_rate": round(
                unseen_anomaly_hits / max(unseen_count, 1), 4
            ) if unseen_count else 0.0,
        }
        return (
            np.array(y_true),
            np.array(y_pred),
            np.array(anomaly_scores),
            explanations,
            unseen_stats,
        )

    def evaluate_metrics(self, y_true, y_pred):
        if len(y_true) == 0:
            return {
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1_Score": 0.0,
            }

        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    def generate_explanation_json(
        self,
        time_step,
        current_state,
        incoming_pattern,
        is_unseen,
        mapped_state,
        path_prob,
        decision,
        transitions=None,
        distance=0,
        confidence=None,
    ):
        confidence_label = "low" if decision == "ANOMALY" else "high"
        return {
            "time_step": time_step,
            "state": current_state,
            "pattern": incoming_pattern,
            "status": "unseen" if is_unseen else "known",
            "mapped_to": mapped_state if is_unseen else incoming_pattern,
            "distance": distance,
            "transitions": transitions or [],
            "probability": round(path_prob, 4),
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "confidence_label": confidence_label,
            "decision": decision.lower(),
        }

    def analyze_counterfactual(self, current_state, original_pattern, alternative_pattern):
        print(
            f"\n[Counterfactual Analysis] Orijinal: '{original_pattern}' | "
            f"Alternatif: '{alternative_pattern}'"
        )

        prob_orig, _ = self.calculate_path_probability([current_state, original_pattern])
        decision_orig, conf_orig = self.evaluate_confidence(prob_orig)

        prob_alt, _ = self.calculate_path_probability([current_state, alternative_pattern])
        decision_alt, conf_alt = self.evaluate_confidence(prob_alt)

        print(f" -> Orijinal Karar: {decision_orig} (Güven: {conf_orig})")
        print(f" -> Alternatif Karar: {decision_alt} (Güven: {conf_alt})")

        return {
            "original_decision": decision_orig,
            "alternative_decision": decision_alt,
            "probability_change": round(prob_alt - prob_orig, 4),
        }
