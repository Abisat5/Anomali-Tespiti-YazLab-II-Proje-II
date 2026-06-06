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
from src.utils.metrics import MetricCalculator
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

    def __init__(self, config):
        self.config = config
        self.group_id = self.config["project"]["group_id"]
        self.random_seeds = self.config["experiment_settings"]["random_seeds"]
        self.datasets = self.config["project"]["datasets"]
        self.seq_len = self.config["deep_learning"]["sequence_length"]
        self.threshold_percentile = self.config["deep_learning"]["anomaly_threshold_percentile"]

        self.data_loader = DataLoader(config)
        self.preprocessor = Preprocessor()
        self.noise_injector = NoiseInjector(config)
        self.visualizer = Visualizer(config)
        self.stat_analyzer = StatisticalAnalyzer()
        self.logger = ExperimentLogger(config)
        self.metric_calc = MetricCalculator(self.logger)

        self.seed_metrics = {
            "LSTM": [],
            "CNN1D": [],
            "Automata": [],
        }
        self.skab_fold_metrics = {
            "LSTM": [],
            "CNN1D": [],
            "Automata": [],
        }
        self._parameter_context = None

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

        print(f"[Pipeline] {dataset_name} etiket sütunu: {label_col}")
        print(f"[Pipeline] {dataset_name} özellik sayısı: {len(feature_cols)}")

        if dataset_name == "SKAB":
            self._run_skab_groupkfold(df, feature_cols, label_col)
        else:
            self._run_batadal_chronological(df, feature_cols, label_col)

    def _run_skab_groupkfold(self, df, feature_cols, label_col):
        folds = self.data_loader.get_group_kfold_splits(df, label_col)

        for seed in self.random_seeds:
            print(f"\n>>> [SKAB] Seed {seed} deneyleri başlıyor <<<")
            set_deterministic_seed(seed)

            fold_lstm, fold_cnn, fold_auto = [], [], []

            for fold_data in folds:
                fold_idx = fold_data["fold"]
                print(f"\n[SKAB] Fold {fold_idx}/{len(folds)} işleniyor...")

                context = self._build_context(
                    dataset_name="SKAB",
                    seed=seed,
                    fold_idx=fold_idx,
                    train_df=fold_data["train"],
                    val_df=fold_data["val"],
                    test_df=fold_data["test"],
                    feature_cols=feature_cols,
                    label_col=label_col,
                )
                self._parameter_context = context

                fold_results = self._run_fold_scenarios(context)
                fold_lstm.append(fold_results["LSTM"]["original"]["F1_Score"])
                fold_cnn.append(fold_results["CNN1D"]["original"]["F1_Score"])
                fold_auto.append(fold_results["Automata"]["original"]["F1_Score"])

            self.seed_metrics["LSTM"].append(float(np.mean(fold_lstm)))
            self.seed_metrics["CNN1D"].append(float(np.mean(fold_cnn)))
            self.seed_metrics["Automata"].append(float(np.mean(fold_auto)))

            self.logger.log_aggregated_metrics(
                f"SKAB_seed_{seed}_LSTM_mean_f1",
                self.stat_analyzer.calculate_mean_and_std(
                    [{"F1_Score": v} for v in fold_lstm]
                ),
            )

    def _run_batadal_chronological(self, df, feature_cols, label_col):
        for seed in self.random_seeds:
            print(f"\n>>> [BATADAL] Seed {seed} deneyleri başlıyor <<<")
            set_deterministic_seed(seed)

            train_df, val_df, test_df = self.data_loader.split_data(df, "BATADAL")
            context = self._build_context(
                dataset_name="BATADAL",
                seed=seed,
                fold_idx=None,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_cols=feature_cols,
                label_col=label_col,
            )
            self._parameter_context = context
            self._run_fold_scenarios(context)

    def _build_context(
        self,
        dataset_name,
        seed,
        fold_idx,
        train_df,
        val_df,
        test_df,
        feature_cols,
        label_col,
    ):
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
        label_col = context["label_col"]

        if scenario == "noisy":
            for frame in (train_df, val_df, test_df):
                frame[feature_cols] = self.noise_injector.inject_noise(
                    frame[feature_cols].values
                )

        return train_df, val_df, test_df

    def _preprocess_frames(self, train_df, val_df, test_df, feature_cols, label_col):
        train_scaled, val_scaled, test_scaled, used_features = self.preprocessor.normalize(
            train_df, val_df, test_df, feature_cols
        )
        train_pc1, val_pc1, test_pc1 = self.preprocessor.apply_pca(
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
        scenario_results = {"LSTM": {}, "CNN1D": {}, "Automata": {}}

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
                print("[Pipeline] Unseen senaryosu otomata odaklıdır; DL modelleri atlanıyor.")
                dl_results = {
                    "LSTM": self._empty_metrics(),
                    "CNN1D": self._empty_metrics(),
                }
            else:
                dl_results = self._run_deep_learning(context, scenario, processed)

            automata_results = self._run_automata(context, scenario, processed)

            scenario_results["LSTM"][scenario] = dl_results["LSTM"]
            scenario_results["CNN1D"][scenario] = dl_results["CNN1D"]
            scenario_results["Automata"][scenario] = automata_results

        return scenario_results

    def _empty_metrics(self):
        return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0}

    def _run_deep_learning(self, context, scenario, processed):
        results = {}
        dataset_name = context["dataset_name"]
        fold_idx = context["fold_idx"]

        train_x = processed["train_scaled"][processed["feature_cols"]].values
        val_x = processed["val_scaled"][processed["feature_cols"]].values
        test_x = processed["test_scaled"][processed["feature_cols"]].values

        x_train, y_train = create_sequences(train_x, processed["train_labels"], self.seq_len)
        x_val, y_val = create_sequences(val_x, processed["val_labels"], self.seq_len)
        x_test, y_test = create_sequences(test_x, processed["test_labels"], self.seq_len)

        if len(x_train) == 0 or len(x_test) == 0:
            empty = self._empty_metrics()
            return {"LSTM": empty, "CNN1D": empty}

        n_features = x_train.shape[2]
        model_builders = {
            "LSTM": LSTMAnomalyDetector,
            "CNN1D": CNN1DAnomalyDetector,
        }

        for model_name, builder in model_builders.items():
            detector = builder(self.config, self.seq_len, n_features)
            trainer = ModelTrainer(self.config, detector)
            trainer.setup_training()
            trainer.train(x_train, x_val)

            threshold = threshold_from_validation(
                detector.model, x_val, self.threshold_percentile
            )
            y_pred, scores = predict_with_threshold(detector.model, x_test, threshold)

            metrics = {
                "Accuracy": float(np.mean(y_pred == y_test)),
                "Precision": float(self._safe_metric(y_test, y_pred, "precision")),
                "Recall": float(self._safe_metric(y_test, y_pred, "recall")),
                "F1_Score": float(self._safe_metric(y_test, y_pred, "f1")),
            }

            metric_key = self._metric_key(context, model_name, scenario)
            self.logger.log_metrics(metric_key, metrics)

            if dataset_name == "SKAB" and fold_idx is not None and scenario == "original":
                self.logger.log_fold_metrics(dataset_name, fold_idx, model_name, metrics)
                self.skab_fold_metrics[model_name].append(metrics["F1_Score"])
            elif dataset_name == "BATADAL" and scenario == "original":
                self.seed_metrics[model_name].append(metrics["F1_Score"])

            if scenario == "original":
                self.visualizer.plot_confusion_matrix(
                    y_test, y_pred, metric_key.replace(" ", "_")
                )
                self.visualizer.plot_roc_curve(
                    y_test, scores, metric_key.replace(" ", "_")
                )

            results[model_name] = metrics
            print(f"[Pipeline] {model_name} / {scenario}: {metrics}")

        return results

    def _run_automata(self, context, scenario, processed):
        dataset_name = context["dataset_name"]
        seed = context["seed"]
        fold_idx = context["fold_idx"]

        sax_converter = SAXConverter(self.config)
        sax_converter.fit_transform(processed["train_pc1"])

        automata = ProbabilisticAutomata(self.config, sax_converter)
        automata.fit(processed["train_pc1"])

        use_unseen_mapping = scenario in ("original", "noisy", "unseen")
        y_true, y_pred, explanations = automata.predict_from_pc1(
            processed["test_pc1"],
            processed["test_labels"],
            use_unseen_mapping=use_unseen_mapping,
        )

        metrics = automata.evaluate_metrics(y_true, y_pred)
        metric_key = self._metric_key(context, "Automata", scenario)
        self.logger.log_metrics(metric_key, metrics)

        if dataset_name == "SKAB" and fold_idx is not None and scenario == "original":
            self.logger.log_fold_metrics(dataset_name, fold_idx, "Automata", metrics)
            self.skab_fold_metrics["Automata"].append(metrics["F1_Score"])
        elif dataset_name == "BATADAL" and scenario == "original":
            self.seed_metrics["Automata"].append(metrics["F1_Score"])

        max_logs = self.config["explainability"].get("max_logged_decisions", 50)
        for explanation in explanations[:max_logs]:
            self.logger.log_explainability(explanation)

        if scenario == "original":
            self.visualizer.plot_transition_heatmap(automata.transition_probabilities)
            self.visualizer.plot_state_diagram(automata.transition_probabilities)

            if len(y_true) > 0:
                self.visualizer.plot_confusion_matrix(
                    y_true, y_pred, metric_key.replace(" ", "_")
                )

            if explanations:
                sample = explanations[0]
                alt_pattern = sample["mapped_to"]
                if sample["status"] == "unseen":
                    automata.analyze_counterfactual(
                        sample["state"], sample["pattern"], alt_pattern
                    )

        print(f"[Pipeline] Automata / {scenario}: {metrics}")
        return metrics

    def _metric_key(self, context, model_name, scenario):
        parts = [
            context["dataset_name"],
            f"seed_{context['seed']}",
        ]
        if context["fold_idx"] is not None:
            parts.append(f"fold_{context['fold_idx']}")
        parts.extend([model_name, scenario])
        return "_".join(parts)

    def _safe_metric(self, y_true, y_pred, metric_name):
        from sklearn.metrics import f1_score, precision_score, recall_score

        if metric_name == "precision":
            return precision_score(y_true, y_pred, zero_division=0)
        if metric_name == "recall":
            return recall_score(y_true, y_pred, zero_division=0)
        return f1_score(y_true, y_pred, zero_division=0)

    def _run_statistical_analysis(self):
        print("\n[Pipeline] İstatistiksel analiz başlatılıyor...")

        for model_name, scores in self.seed_metrics.items():
            if not scores:
                continue
            aggregated = self.stat_analyzer.calculate_mean_and_std(
                [{"F1_Score": score} for score in scores]
            )
            self.logger.log_aggregated_metrics(f"{model_name}_seed_f1_summary", aggregated)
            print(f"[Pipeline] {model_name} seed özeti: {aggregated}")

        for model_name, fold_scores in self.skab_fold_metrics.items():
            if not fold_scores:
                continue
            aggregated = self.stat_analyzer.calculate_mean_and_std(
                [{"F1_Score": score} for score in fold_scores]
            )
            self.logger.log_aggregated_metrics(f"SKAB_{model_name}_fold_f1_summary", aggregated)
            print(f"[Pipeline] SKAB {model_name} fold özeti: {aggregated}")

        if len(self.seed_metrics["LSTM"]) >= 5 and len(self.seed_metrics["CNN1D"]) >= 5:
            self.stat_analyzer.run_wilcoxon_test(
                self.seed_metrics["LSTM"],
                self.seed_metrics["CNN1D"],
            )

    def _run_parameter_analysis(self):
        if self._parameter_context is None:
            print("[Pipeline] Parametre analizi için bağlam bulunamadı, atlanıyor.")
            return

        print("\n[Pipeline] Parametre duyarlılık analizi başlatılıyor...")
        context = self._parameter_context
        train_df, val_df, test_df = self._apply_scenario_to_frames(context, "original")
        processed = self._preprocess_frames(
            train_df,
            val_df,
            test_df,
            context["feature_cols"],
            context["label_col"],
        )

        sax_converter = SAXConverter(self.config)
        sax_symbols = sax_converter.fit_transform(processed["train_pc1"])
        automata = ProbabilisticAutomata(self.config, sax_converter)

        param_analyzer = ParameterAnalyzer(self.config, automata)
        window_results = param_analyzer.analyze_window_size(sax_symbols)
        alphabet_results = param_analyzer.analyze_alphabet_size(
            processed["train_pc1"], SAXConverter
        )

        all_results = window_results + alphabet_results
        self.logger.log_parameter_analysis(all_results)
        param_analyzer.export_analysis_to_table(all_results)

        self.visualizer.plot_parameter_sensitivity(
            window_results, "Value", "State_Count", "Window Size vs State Count"
        )
        self.visualizer.plot_parameter_sensitivity(
            alphabet_results, "Value", "State_Count", "Alphabet Size vs State Count"
        )

    def _save_summary(self):
        print(f"\n[Pipeline] Deney logları kaydedildi: {self.logger.log_file}")
