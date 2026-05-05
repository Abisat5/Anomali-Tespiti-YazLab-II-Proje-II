import unittest
import sys
import os

# src klasörünün import edilebilmesi için sistem yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.automata import ProbabilisticAutomata

class MockConfig:
    def __init__(self):
        self.config = {
            'automata': {
                'variation_params': {'window_sizes': [3, 4, 5, 6]}
            }
        }
    def __getitem__(self, key):
        return self.config[key]

class TestAutomataUnseen(unittest.TestCase):
    def setUp(self):
        """Test ortamını hazırla."""
        self.config = MockConfig()
        self.automata = ProbabilisticAutomata(self.config, sax_converter=None)
        
        # Sahte bir eğitim sözlüğü (transition modeli) oluştur
        self.automata.transition_probabilities = {
            "abc": {"bcd": 1.0},
            "def": {"efg": 1.0},
            "ghi": {"hij": 1.0}
        }

    def test_levenshtein_distance(self):
        """Levenshtein algoritmasının doğru hesaplayıp hesaplamadığını test et."""
        dist = self.automata.calculate_levenshtein_distance("abc", "abd")
        self.assertEqual(dist, 1) # Sadece 'c' harfi 'd' oldu (1 işlem)
        
        dist2 = self.automata.calculate_levenshtein_distance("abc", "xyz")
        self.assertEqual(dist2, 3) # Tamamen farklı (3 işlem)

    def test_handle_unseen_pattern(self):
        """Unseen pattern'ın en yakın duruma eşlenip eşlenmediğini test et."""
        unseen = "abx"
        closest, distance = self.automata.handle_unseen_pattern(unseen)
        
        # 'abx', sözlükteki 'abc'ye en yakındır (Mesafe: 1)
        self.assertEqual(closest, "abc")
        self.assertEqual(distance, 1)

if __name__ == '__main__':
    unittest.main()