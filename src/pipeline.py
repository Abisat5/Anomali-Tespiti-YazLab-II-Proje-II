import numpy as np

from src.models.automata import ProbabilisticAutomata
from src.models.deep_learning import CNN1DAnomalyDetector, LSTMAnomalyDetector
from src.models.trainer import ModelTrainer
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.noise_injector import NoiseInjector
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.sax_converter import SAXConverter
from src.utils.analyzer import ParameterAnalyzer
from src.utils.logger import ExperimentLogger
from src.utils.seed_manager import set_deterministic_seed
from src.utils.sequence_builder import (
    create_sequences,
    predict_with_threshold,
    threshold_from_validation,
)
from src.utils.statistics import StatisticalAnalyzer
from src.utils.visualizer import Visualizer


class Pipeline:
    SCENARIOS = ("original", "noisy", "unseen")
    MODELS = ("LSTM", "CNN1D", "Automata")

    def __init__(self, config):
        self.config = config
        self.group_id = self.config["project"]["group_id"]
        self.datasets = self.config["project"]["datasets"]
        self.seq_len = self.config["deep_learning"]["sequence_length"]
        self.threshold_percentile = self.config["deep_learning"]["anomaly_threshold_percentile"]

        quick = self.config.get("quick_test", {})
        if quick.get("enabled", False):
            self.random_seeds = quick.get("seeds", [42])
            self.skab_cv_folds = quick.get("skab_folds", 2)
            self.config["deep_learning"]["epochs"] = quick.get("epochs", 3)
            print("[Pipeline] quick_test modu aktif.")
        else:
            self.random_seeds = self.config["experiment_settings"]["random_seeds"]
            self.skab_cv_folds = self.config["experiment_settings"].get("skab_cv_folds", 5)

        self.data_loader = DataLoader(config)
        self.noise_injector = NoiseInjector(config)
        self.visualizer = Visualizer(config)
        self.stat_analyzer = StatisticalAnalyzer()
        self.logger = ExperimentLogger(config)

        self.skab_seed_metrics = {model: [] for model in self.MODELS}
        self.batadal_seed_metrics = {model: [] for model in self.MODELS}
        self.skab_fold_metrics = {
            model: {s: [] for s in self.SCENARIOS} for model in self.MODELS
        }
        self.scenario_metrics = {
            dataset: {s: [] for s in self.SCENARIOS} for dataset in self.datasets
        }
        self._parameter_contexts = {}
        self._batadal_predictions_by_seed = {}

    def run(self):
        print(f"\n{'=' * 50}")
        print(f"--- {self.group_id}. Grup Pipeline Akışı Başlıyor ---")
        print(f"{'=' * 50}")

        for dataset_name in self.datasets:
            self._run_dataset(dataset_name)

        self._run_statistical_analysis()
        self._run_parameter_analysis()
        self._save_summary()

        print("\n--- Pipeline Akışı Başarıyla Tamamlandı ---\n")

    def _run_dataset(self, dataset_name):
        df = self.data_loader.load_data(dataset_name)
        if df is None:
            print(f"[Pipeline] {dataset_name} verisi bulunamadı, atlanıyor.")
            return

        label_col = self.data_loader.detect_label_column(df, dataset_name)
        feature_cols = self.data_loader.get_feature_columns(df, label_col)
        df = self.data_loader.handle_missing_values(df, feature_cols, label_col)

        self.logger.log_dataset_info(
            dataset_name,
            {
                "label_column": label_col,
                "feature_count": len(feature_cols),
                "feature_columns": feature_cols,
                "row_count": len(df),
            },
        )
        print(f"[Pipeline] {dataset_name} etiket sütunu: {label_col}")

        if dataset_name == "SKAB":
            self._run_skab_groupkfold(df, feature_cols, label_col)
        else:
            self._run_batadal_chronological(df, feature_cols, label_col)

    def _run_skab_groupkfold(self, df, feature_cols, label_col):
        folds = self.data_loader.get_group_kfold_splits(df, label_col, n_splits=self.skab_cv_folds)

        for seed in self.random_seeds:
            print(f"\n>>> [SKAB] Seed {seed} <<<")
            set_deterministic_seed(seed)
            seed_fold_f1 = {model: [] for model in self.MODELS}

            for fold_data in folds:
                fold_idx = fold_data["fold"]
                context = self._build_context(
                    "SKAB",
                    seed,
                    fold_idx,
                    fold_data["train"],
                    fold_data["val"],
                    fold_data["test"],
                    feature_cols,
                    label_col,
                )
                if "SKAB" not in self._parameter_contexts:
                    self._parameter_contexts["SKAB"] = context

                fold_results = self._run_fold_scenarios(context)

                for model in seed_fold_f1:
                    seed_fold_f1[model].append(fold_results[model]["original"]["F1_Score"])

            for model, scores in seed_fold_f1.items():
                if scores:
                    self.skab_seed_metrics[model].append(float(np.mean(scores)))
                    self.logger.log_aggregated_metrics(
                        f"SKAB_seed_{seed}_{model}_fold_mean_f1",
                        self.stat_analyzer.calculate_numeric_mean_std(scores),
                    )

    def _run_batadal_chronological(self, df, feature_cols, label_col):
        for seed in self.random_seeds:
            print(f"\n>>> [BATADAL] Seed {seed} <<<")
            set_deterministic_seed(seed)

            train_df, val_df, test_df = self.data_loader.split_data(df, "BATADAL")
            context = self._build_context(
                "BATADAL",
                seed,
                None,
                train_df,
                val_df,
                test_df,
                feature_cols,
                label_col,
            )
            if "BATADAL" not in self._parameter_contexts:
                self._parameter_contexts["BATADAL"] = context

            self._batadal_predictions_by_seed[seed] = {}
            self._run_fold_scenarios(context)

    def _build_context(self, dataset_name, seed, fold_idx, train_df, val_df, test_df, feature_cols, label_col):
        return {
            "dataset_name": dataset_name,
            "seed": seed,
            "fold_idx": fold_idx,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "feature_cols": feature_cols,
            "label_col": label_col,
        }

    def _apply_scenario_to_frames(self, context, scenario):
        train_df = context["train_df"].copy()
        val_df = context["val_df"].copy()
        test_df = context["test_df"].copy()
        feature_cols = context["feature_cols"]

        if scenario == "noisy":
            for frame in (train_df, val_df, test_df):
                frame[feature_cols] = self.noise_injector.inject_noise(frame[feature_cols].values)

        return train_df, val_df, test_df

    def _preprocess_frames(self, train_df, val_df, test_df, feature_cols, label_col):
        preprocessor = Preprocessor()
        train_scaled, val_scaled, test_scaled, used_features = preprocessor.normalize(
            train_df, val_df, test_df, feature_cols
        )
        train_pc1, val_pc1, test_pc1 = preprocessor.apply_pca(
            train_scaled, val_scaled, test_scaled, used_features
        )
        return {
            "train_scaled": train_scaled,
            "val_scaled": val_scaled,
            "test_scaled": test_scaled,
            "feature_cols": used_features,
            "train_pc1": train_pc1,
            "val_pc1": val_pc1,
            "test_pc1": test_pc1,
            "train_labels": train_df[label_col].values,
            "val_labels": val_df[label_col].values,
            "test_labels": test_df[label_col].values,
        }

    def _run_fold_scenarios(self, context):
        scenario_results = {model: {} for model in self.MODELS}

        for scenario in self.SCENARIOS:
            print(
                f"\n[Pipeline] Senaryo: {scenario.upper()} | "
                f"{context['dataset_name']} | Seed: {context['seed']}"
            )

            train_df, val_df, test_df = self._apply_scenario_to_frames(context, scenario)
            processed = self._preprocess_frames(
                train_df,
                val_df,
                test_df,
                context["feature_cols"],
                context["label_col"],
            )

            if scenario == "unseen":
                print("[Pipeline] Unseen senaryosu: otomata odaklı, DL atlanıyor.")
                dl_results = {m: self._empty_metrics() for m in ("LSTM", "CNN1D")}
            else:
                dl_results = self._run_deep_learning(context, scenario, processed)

            automata_results, unseen_stats = self._run_automata(context, scenario, processed)

            for model in ("LSTM", "CNN1D"):
                scenario_results[model][scenario] = dl_results[model]
            scenario_results["Automata"][scenario] = automata_results

            summary = {
                "LSTM": dl_results["LSTM"],
                "CNN1D": dl_results["CNN1D"],
                "Automata": automata_results,
                "unseen_stats": unseen_stats,
            }
            self.logger.log_scenario_summary(
                context["dataset_name"], context["seed"], scenario, summary
            )
            self.scenario_metrics[context["dataset_name"]][scenario].append(automata_results)

        return scenario_results

    def _empty_metrics(self):
        return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0}

    def _run_deep_learning(self, context, scenario, processed):
        results = {}
        dataset_name = context["dataset_name"]

        train_x = processed["train_scaled"][processed["feature_cols"]].values
        val_x = processed["val_scaled"][processed["feature_cols"]].values
        test_x = processed["test_scaled"][processed["feature_cols"]].values

        x_train, _ = create_sequences(train_x, processed["train_labels"], self.seq_len)
        x_val, _ = create_sequences(val_x, processed["val_labels"], self.seq_len)
        x_test, y_test = create_sequences(test_x, processed["test_labels"], self.seq_len)

        if len(x_train) == 0 or len(x_test) == 0:
            return {m: self._empty_metrics() for m in ("LSTM", "CNN1D")}

        n_features = x_train.shape[2]
        builders = {"LSTM": LSTMAnomalyDetector, "CNN1D": CNN1DAnomalyDetector}

        for model_name, builder in builders.items():
            detector = builder(self.config, self.seq_len, n_features)
            trainer = ModelTrainer(self.config, detector)
            trainer.setup_training()
            trainer.train(x_train, x_val)

            threshold = threshold_from_validation(detector.model, x_val, self.threshold_percentile)
            y_pred, scores = predict_with_threshold(detector.model, x_test, threshold)
            metrics = self._compute_metrics(y_test, y_pred)

            metric_key = self._metric_key(context, model_name, scenario)
            self.logger.log_metrics(metric_key, metrics)
            self._record_metrics(context, model_name, scenario, metrics)

            safe_key = metric_key.replace(" ", "_")
            self.visualizer.plot_confusion_matrix(y_test, y_pred, safe_key)
            self.visualizer.plot_roc_curve(y_test, scores, safe_key)
            self.visualizer.plot_pr_curve(y_test, scores, safe_key)

            if dataset_name == "BATADAL" and scenario == "original":
                self._batadal_predictions_by_seed[context["seed"]][model_name] = (
                    y_test.copy(),
                    y_pred.copy(),
                )
                self.logger.log_batadal_test_result(context["seed"], scenario, model_name, metrics)

            results[model_name] = metrics
            print(f"[Pipeline] {model_name}/{scenario}: {metrics}")

        return results

    def _run_automata(self, context, scenario, processed):
        sax_converter = SAXConverter(self.config)
        sax_converter.fit_transform(processed["train_pc1"])

        automata = ProbabilisticAutomata(self.config, sax_converter)
        automata.fit(processed["train_pc1"])

        y_true, y_pred, scores, explanations, unseen_stats = automata.predict_from_pc1(
            processed["test_pc1"],
            processed["test_labels"],
            use_unseen_mapping=True,
        )

        metrics = automata.evaluate_metrics(y_true, y_pred)
        metric_key = self._metric_key(context, "Automata", scenario)
        self.logger.log_metrics(metric_key, metrics)
        self._record_metrics(context, "Automata", scenario, metrics)

        max_logs = self.config["explainability"].get("max_logged_decisions", 50)
        for explanation in explanations[:max_logs]:
            self.logger.log_explainability(explanation)

        safe_key = metric_key.replace(" ", "_")
        self.visualizer.plot_confusion_matrix(y_true, y_pred, safe_key)
        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            self.visualizer.plot_roc_curve(y_true, scores, safe_key)
            self.visualizer.plot_pr_curve(y_true, scores, safe_key)

        viz_suffix = f"_{context['dataset_name']}_seed{context['seed']}"
        if scenario == "original":
            self.visualizer.plot_transition_heatmap(
                automata.transition_probabilities, suffix=viz_suffix
            )
            self.visualizer.plot_state_diagram(
                automata.transition_probabilities, suffix=viz_suffix
            )

        if scenario == "unseen" and explanations:
            sample = next((e for e in explanations if e["status"] == "unseen"), explanations[0])
            automata.analyze_counterfactual(sample["state"], sample["pattern"], sample["mapped_to"])

        if context["dataset_name"] == "BATADAL":
            self.logger.log_batadal_test_result(context["seed"], scenario, "Automata", metrics)
            if scenario == "original" and len(y_true) > 0:
                self._batadal_predictions_by_seed[context["seed"]]["Automata"] = (
                    y_true.copy(),
                    y_pred.copy(),
                )

        print(f"[Pipeline] Automata/{scenario}: {metrics} | unseen={unseen_stats}")
        return metrics, unseen_stats

    def _record_metrics(self, context, model_name, scenario, metrics):
        dataset_name = context["dataset_name"]
        fold_idx = context["fold_idx"]

        if dataset_name == "SKAB" and fold_idx is not None:
            self.logger.log_fold_metrics(dataset_name, fold_idx, model_name, scenario, metrics)
            self.skab_fold_metrics[model_name][scenario].append(metrics["F1_Score"])
        elif dataset_name == "BATADAL" and scenario == "original":
            self.batadal_seed_metrics[model_name].append(metrics["F1_Score"])

    def _metric_key(self, context, model_name, scenario):
        parts = [context["dataset_name"], f"seed_{context['seed']}"]
        if context["fold_idx"] is not None:
            parts.append(f"fold_{context['fold_idx']}")
        parts.extend([model_name, scenario])
        return "_".join(parts)

    def _compute_metrics(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    def _run_statistical_analysis(self):
        print("\n[Pipeline] İstatistiksel analiz başlatılıyor...")

        for dataset_prefix, seed_metrics in (
            ("SKAB", self.skab_seed_metrics),
            ("BATADAL", self.batadal_seed_metrics),
        ):
            for model_name, scores in seed_metrics.items():
                if not scores:
                    continue
                agg = self.stat_analyzer.calculate_mean_and_std(
                    [{"F1_Score": s} for s in scores]
                )
                key = f"{dataset_prefix}_{model_name}_seed_f1_summary"
                self.logger.log_aggregated_metrics(key, agg)
                print(f"[Pipeline] {key}: {agg}")

        for model_name, scenario_dict in self.skab_fold_metrics.items():
            for scenario, scores in scenario_dict.items():
                if not scores:
                    continue
                agg = self.stat_analyzer.calculate_numeric_mean_std(scores)
                key = f"SKAB_{model_name}_{scenario}_fold_f1_summary"
                self.logger.log_aggregated_metrics(key, agg)
                print(f"[Pipeline] {key}: {agg}")

        for dataset_prefix, seed_metrics in (
            ("SKAB", self.skab_seed_metrics),
            ("BATADAL", self.batadal_seed_metrics),
        ):
            comparisons = [
                ("LSTM", "CNN1D"),
                ("LSTM", "Automata"),
                ("CNN1D", "Automata"),
            ]
            for name_a, name_b in comparisons:
                scores_a = seed_metrics[name_a]
                scores_b = seed_metrics[name_b]
                if len(scores_a) >= 2 and len(scores_b) >= 2:
                    result = self.stat_analyzer.run_wilcoxon_test(
                        scores_a,
                        scores_b,
                        label=f"{dataset_prefix}_{name_a}_vs_{name_b}",
                    )
                    self.logger.log_statistical_test(result)

        mcnemar_pairs = [
            ("LSTM", "CNN1D"),
            ("LSTM", "Automata"),
            ("CNN1D", "Automata"),
        ]
        for seed, preds in self._batadal_predictions_by_seed.items():
            for name_a, name_b in mcnemar_pairs:
                if name_a in preds and name_b in preds:
                    y_true, y_pred_a = preds[name_a]
                    _, y_pred_b = preds[name_b]
                    result = self.stat_analyzer.run_mcnemar_test(
                        y_true,
                        y_pred_a,
                        y_pred_b,
                        label=f"BATADAL_seed_{seed}_{name_a}_vs_{name_b}",
                    )
                    self.logger.log_statistical_test(result)

    def _run_parameter_analysis(self):
        if not self._parameter_contexts:
            print("[Pipeline] Parametre analizi atlanıyor (bağlam yok).")
            return

        all_results = []

        for dataset_name, context in self._parameter_contexts.items():
            print(f"\n[Pipeline] Parametre duyarlılık analizi: {dataset_name}")
            train_df, val_df, test_df = self._apply_scenario_to_frames(context, "original")
            processed = self._preprocess_frames(
                train_df,
                val_df,
                test_df,
                context["feature_cols"],
                context["label_col"],
            )

            sax_converter = SAXConverter(self.config)
            sax_converter.fit_transform(processed["train_pc1"])
            automata = ProbabilisticAutomata(self.config, sax_converter)
            suffix = f"_{dataset_name.lower()}"

            param_analyzer = ParameterAnalyzer(self.config, automata)
            window_results = param_analyzer.analyze_window_size(
                processed["train_pc1"],
                processed["test_pc1"],
                processed["test_labels"],
                sax_converter,
            )
            alphabet_results = param_analyzer.analyze_alphabet_size(
                processed["train_pc1"],
                processed["test_pc1"],
                processed["test_labels"],
                SAXConverter,
            )

            for row in window_results + alphabet_results:
                row["Dataset"] = dataset_name
            all_results.extend(window_results + alphabet_results)

            param_analyzer.export_analysis_to_table(
                window_results + alphabet_results,
                filepath=f"logs/parameter_analysis_{dataset_name.lower()}.csv",
            )

            self.visualizer.plot_parameter_sensitivity(
                window_results, "Value", "State_Count",
                f"{dataset_name} Window Size vs State Count", suffix=suffix,
            )
            self.visualizer.plot_parameter_sensitivity(
                window_results, "Value", "F1_Score",
                f"{dataset_name} Window Size vs F1 Score", suffix=suffix,
            )
            self.visualizer.plot_parameter_sensitivity(
                window_results, "Value", "Transition_Density",
                f"{dataset_name} Window Size vs Transition Density", suffix=suffix,
            )
            self.visualizer.plot_parameter_sensitivity(
                alphabet_results, "Value", "State_Count",
                f"{dataset_name} Alphabet Size vs State Count", suffix=suffix,
            )
            self.visualizer.plot_parameter_sensitivity(
                alphabet_results, "Value", "F1_Score",
                f"{dataset_name} Alphabet Size vs F1 Score", suffix=suffix,
            )
            self.visualizer.plot_parameter_sensitivity(
                alphabet_results, "Value", "Transition_Density",
                f"{dataset_name} Alphabet Size vs Transition Density", suffix=suffix,
            )

        self.logger.log_parameter_analysis(all_results)

    def _save_summary(self):
        self.logger.save()
        self.logger.export_summary_csv()
        self.logger.export_explainability()
        print(f"\n[Pipeline] Deney logları: {self.logger.log_file}")
