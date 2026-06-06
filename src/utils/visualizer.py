import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.log_dir = config["paths"]["logs"]
        os.makedirs(self.log_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        from sklearn.metrics import confusion_matrix

        print(f"\n[Visualizer] {model_name} için Confusion Matrix oluşturuluyor...")

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
        )

        plt.title(f"{model_name} - Confusion Matrix")
        plt.ylabel("Gerçek Etiket")
        plt.xlabel("Tahmin Edilen Etiket")

        save_path = os.path.join(self.log_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Grafik kaydedildi: {save_path}")

    def plot_roc_curve(self, y_true, y_scores, model_name):
        from sklearn.metrics import auc, roc_curve

        print(f"[Visualizer] {model_name} için ROC eğrisi oluşturuluyor...")

        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} - ROC Eğrisi")
            plt.legend(loc="lower right")

            save_path = os.path.join(self.log_dir, f"{model_name}_roc_curve.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"[Visualizer] Grafik kaydedildi: {save_path}")
        except Exception as e:
            print(f"[Hata] ROC eğrisi çizilemedi: {e}")

    def plot_pr_curve(self, y_true, y_scores, model_name):
        from sklearn.metrics import average_precision_score, precision_recall_curve

        print(f"[Visualizer] {model_name} için PR eğrisi oluşturuluyor...")

        try:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap_score = average_precision_score(y_true, y_scores)

            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, color="teal", lw=2, label=f"AP = {ap_score:.2f}")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"{model_name} - Precision-Recall Eğrisi")
            plt.legend(loc="lower left")

            save_path = os.path.join(self.log_dir, f"{model_name}_pr_curve.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"[Visualizer] Grafik kaydedildi: {save_path}")
        except Exception as e:
            print(f"[Hata] PR eğrisi çizilemedi: {e}")

    def plot_transition_heatmap(self, transition_probs, top_n=None, suffix=""):
        top_n = top_n or self.config["automata"]["visualization"].get("heatmap_top_n", 20)
        print(f"[Visualizer] Automata Transition Heatmap oluşturuluyor (Top {top_n} durum)...")

        df_trans = pd.DataFrame(transition_probs).fillna(0)

        if len(df_trans) > top_n:
            top_states = df_trans.sum(axis=1).sort_values(ascending=False).head(top_n).index
            df_trans = df_trans.loc[top_states, top_states].fillna(0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(df_trans, cmap="YlGnBu", annot=False, cbar_kws={"label": "Geçiş Olasılığı"})
        plt.title("Olasılıksal Otomata - Transition Probability Heatmap")
        plt.xlabel("Sonraki Durum")
        plt.ylabel("Mevcut Durum")

        save_path = os.path.join(self.log_dir, f"automata_transition_heatmap{suffix}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Grafik kaydedildi: {save_path}")

    def plot_state_diagram(self, transition_probs, threshold=None, suffix=""):
        import networkx as nx

        threshold = threshold or self.config["automata"]["visualization"].get(
            "state_diagram_threshold", 0.1
        )
        print(f"[Visualizer] Automata State Diagram oluşturuluyor (Threshold: {threshold})...")

        G = nx.DiGraph()

        for current_state, transitions in transition_probs.items():
            for next_state, prob in transitions.items():
                if prob >= threshold:
                    G.add_edge(current_state, next_state, weight=prob)

        if len(G.nodes) == 0:
            print("[Uyarı] State diagram için yeterli geçiş bulunamadı.")
            return

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=0.5, seed=42)

        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, width=1.5)

        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title(f"Automata State Diagram (Geçişler >= {threshold})")
        plt.axis("off")

        save_path = os.path.join(self.log_dir, f"automata_state_diagram{suffix}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Grafik kaydedildi: {save_path}")

    def plot_parameter_sensitivity(self, results_list, param_col, value_col, title, suffix=""):
        print(f"[Visualizer] Parametre Duyarlılık Grafiği oluşturuluyor: {title}...")

        if not results_list:
            print("[Uyarı] Çizilecek veri bulunamadı.")
            return

        results_df = pd.DataFrame(results_list)

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=results_df, x=param_col, y=value_col, marker="o", linewidth=2, color="coral")

        plt.title(title)
        plt.xlabel(param_col.replace("_", " "))
        plt.ylabel(value_col.replace("_", " "))
        plt.grid(True, linestyle="--", alpha=0.7)

        safe_title = title.replace(" ", "_").lower()
        save_path = os.path.join(self.log_dir, f"sensitivity_{safe_title}{suffix}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Grafik kaydedildi: {save_path}")
