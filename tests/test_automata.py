import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.automata import ProbabilisticAutomata


class MockConfig:
    def __init__(self):
        self.config = {
            "automata": {
                "base_params": {
                    "window_size": 4,
                    "alphabet_size": 3,
                    "paa_segment_size": 1,
                    "path_probability_threshold": 0.05,
                    "smoothing_alpha": 1.0,
                    "unknown_transition_prob": 0.001,
                },
                "variation_params": {"window_sizes": [3, 4, 5, 6]},
            }
        }

    def __getitem__(self, key):
        return self.config[key]


class TestAutomataUnseen(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.automata = ProbabilisticAutomata(self.config, sax_converter=None)
        self.automata.transition_probabilities = {
            "abc": {"bcd": 0.8, "aaa": 0.2},
            "def": {"efg": 1.0},
        }
        self.automata.train_pattern_set = {"abc", "bcd", "def", "efg"}

    def test_levenshtein_distance(self):
        self.assertEqual(self.automata.calculate_levenshtein_distance("abc", "abd"), 1)
        self.assertEqual(self.automata.calculate_levenshtein_distance("abc", "xyz"), 3)

    def test_handle_unseen_pattern(self):
        closest, distance = self.automata.handle_unseen_pattern("abx")
        self.assertEqual(closest, "abc")
        self.assertEqual(distance, 1)

    def test_resolve_pattern_known(self):
        mapped, is_unseen, distance = self.automata.resolve_pattern("abc")
        self.assertFalse(is_unseen)
        self.assertEqual(mapped, "abc")
        self.assertEqual(distance, 0)

    def test_resolve_pattern_unseen(self):
        mapped, is_unseen, distance = self.automata.resolve_pattern("abx")
        self.assertTrue(is_unseen)
        self.assertEqual(mapped, "abc")
        self.assertEqual(distance, 1)

    def test_get_transition_probability_unknown(self):
        prob = self.automata.get_transition_probability("abc", "zzz")
        self.assertEqual(prob, 0.001)

    def test_calculate_path_probability(self):
        path_prob, transitions = self.automata.calculate_path_probability(["abc", "bcd"])
        self.assertAlmostEqual(path_prob, 0.8, places=4)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["from"], "abc")
        self.assertEqual(transitions[0]["to"], "bcd")

    def test_evaluate_confidence(self):
        decision, confidence = self.automata.evaluate_confidence(0.01)
        self.assertEqual(decision, "ANOMALY")
        self.assertAlmostEqual(confidence, 0.01, places=4)

        decision, confidence = self.automata.evaluate_confidence(0.9)
        self.assertEqual(decision, "NORMAL")
        self.assertAlmostEqual(confidence, 0.9, places=4)

    def test_build_transition_model_smoothing(self):
        automata = ProbabilisticAutomata(self.config, sax_converter=None)
        automata.build_transition_model(["abc", "bcd", "abc", "bcd"])
        probs = automata.transition_probabilities["abc"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)
        self.assertTrue(all(prob > 0 for prob in probs.values()))

    def test_generate_explanation_json(self):
        explanation = self.automata.generate_explanation_json(
            time_step=5,
            current_state="abc",
            incoming_pattern="adc",
            is_unseen=True,
            mapped_state="abc",
            path_prob=0.108,
            decision="ANOMALY",
            transitions=[{"from": "abc", "to": "bcd", "probability": 0.72}],
            distance=1,
            confidence=0.108,
        )
        self.assertEqual(explanation["time_step"], 5)
        self.assertEqual(explanation["status"], "unseen")
        self.assertEqual(explanation["decision"], "anomaly")
        self.assertEqual(explanation["confidence_label"], "low")
        self.assertIn("probability", explanation)


if __name__ == "__main__":
    unittest.main()
