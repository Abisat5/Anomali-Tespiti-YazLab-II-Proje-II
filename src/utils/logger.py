import json
import os
from datetime import datetime


class ExperimentLogger:
    def __init__(self, config):
        self.config = config
        self.log_dir = config["paths"]["logs"]
        os.makedirs(self.log_dir, exist_ok=True)

        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"exp_{self.experiment_id}.json")

        self.results = {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "dataset_info": {},
            "metrics": {},
            "fold_results": {},
            "scenario_summaries": {},
            "batadal_test_results": {},
            "aggregated_results": {},
            "statistical_tests": [],
            "explainability": [],
            "parameter_analysis": [],
        }

    def log_dataset_info(self, dataset_name, info):
        self.results["dataset_info"][dataset_name] = info
        self.save()

    def log_metrics(self, model_name, metrics):
        self.results["metrics"][model_name] = metrics
        self.save()

    def log_fold_metrics(self, dataset_name, fold_idx, model_name, scenario, metrics):
        key = f"{dataset_name}_fold_{fold_idx}_{model_name}_{scenario}"
        self.results["fold_results"][key] = metrics
        self.save()

    def log_scenario_summary(self, dataset_name, seed, scenario, summary):
        key = f"{dataset_name}_seed_{seed}_{scenario}"
        self.results["scenario_summaries"][key] = summary
        self.save()

    def log_batadal_test_result(self, seed, scenario, model_name, metrics):
        key = f"seed_{seed}_{model_name}_{scenario}"
        self.results["batadal_test_results"][key] = metrics
        self.save()

    def log_aggregated_metrics(self, key, metrics):
        self.results["aggregated_results"][key] = metrics
        self.save()

    def log_statistical_test(self, test_result):
        self.results["statistical_tests"].append(test_result)
        self.save()

    def log_parameter_analysis(self, analysis_rows):
        self.results["parameter_analysis"].extend(analysis_rows)
        self.save()

    def log_explainability(self, decision_data):
        self.results["explainability"].append(decision_data)
        self.save()

    def save(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)

    def export_summary_csv(self, filepath="logs/experiment_summary.csv"):
        import pandas as pd

        rows = []
        for key, metrics in self.results["metrics"].items():
            row = {"experiment_key": key}
            row.update(metrics)
            rows.append(row)

        if not rows:
            return None

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"[Logger] Özet tablo kaydedildi: {filepath}")
        return df
