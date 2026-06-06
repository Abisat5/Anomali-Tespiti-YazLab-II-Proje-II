import numpy as np


class StatisticalAnalyzer:
    def calculate_mean_and_std(self, metrics_list):
        print("\n[StatAnalyzer] Ortalama ve Standart Sapma hesaplanıyor...")
        aggregated = {}

        if not metrics_list:
            return aggregated

        keys = metrics_list[0].keys()
        for key in keys:
            values = [m[key] for m in metrics_list]
            mean_val = np.mean(values)
            std_val = np.std(values)
            aggregated[key] = f"{mean_val:.4f} ± {std_val:.4f}"

        return aggregated

    def calculate_numeric_mean_std(self, values):
        if not values:
            return {"mean": 0.0, "std": 0.0}
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    def run_wilcoxon_test(self, model1_f1_scores, model2_f1_scores, label="Model"):
        from scipy.stats import wilcoxon

        print(f"\n[StatAnalyzer] Wilcoxon testi: {label}")
        if len(model1_f1_scores) < 2 or len(model2_f1_scores) < 2:
            print("[Uyarı] Wilcoxon için yeterli örnek yok.")
            return {"test": "wilcoxon", "label": label, "p_value": None, "significance": "Yetersiz veri"}

        try:
            stat, p_value = wilcoxon(model1_f1_scores, model2_f1_scores)
            significance = (
                "Anlamlı Fark VAR (p < 0.05)" if p_value < 0.05 else "Anlamlı Fark YOK (p >= 0.05)"
            )
            print(f"Wilcoxon Stat: {stat:.4f}, p-value: {p_value:.4f} -> {significance}")
            return {
                "test": "wilcoxon",
                "label": label,
                "statistic": float(stat),
                "p_value": float(p_value),
                "significance": significance,
            }
        except Exception as exc:
            print(f"[Hata] Wilcoxon uygulanamadı: {exc}")
            return {"test": "wilcoxon", "label": label, "p_value": None, "significance": str(exc)}

    def run_mcnemar_test(self, y_true, y_pred_a, y_pred_b, label="Model"):
        from scipy.stats import chi2

        print(f"\n[StatAnalyzer] McNemar testi: {label}")
        if len(y_true) == 0:
            return {"test": "mcnemar", "label": label, "p_value": None, "significance": "Yetersiz veri"}

        min_len = min(len(y_true), len(y_pred_a), len(y_pred_b))
        if min_len == 0:
            return {"test": "mcnemar", "label": label, "p_value": None, "significance": "Yetersiz veri"}

        if len({len(y_true), len(y_pred_a), len(y_pred_b)}) > 1:
            print(f"[Uyarı] McNemar için uzunluklar hizalanıyor: {len(y_true)}, {len(y_pred_a)}, {len(y_pred_b)} -> {min_len}")
            y_true = y_true[:min_len]
            y_pred_a = y_pred_a[:min_len]
            y_pred_b = y_pred_b[:min_len]

        both_correct = np.sum((y_pred_a == y_true) & (y_pred_b == y_true))
        a_correct_b_wrong = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
        a_wrong_b_correct = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
        both_wrong = np.sum((y_pred_a != y_true) & (y_pred_b != y_true))

        if a_correct_b_wrong + a_wrong_b_correct == 0:
            return {
                "test": "mcnemar",
                "label": label,
                "p_value": 1.0,
                "significance": "Anlamlı Fark YOK (p >= 0.05)",
                "discordant_pairs": 0,
            }

        statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / (
            a_correct_b_wrong + a_wrong_b_correct
        )
        p_value = 1 - chi2.cdf(statistic, df=1)
        significance = (
            "Anlamlı Fark VAR (p < 0.05)" if p_value < 0.05 else "Anlamlı Fark YOK (p >= 0.05)"
        )

        print(
            f"McNemar: b01={a_correct_b_wrong}, b10={a_wrong_b_correct}, "
            f"p-value={p_value:.4f} -> {significance}"
        )
        return {
            "test": "mcnemar",
            "label": label,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significance": significance,
            "discordant_pairs": int(a_correct_b_wrong + a_wrong_b_correct),
            "both_wrong": int(both_wrong),
            "both_correct": int(both_correct),
        }
