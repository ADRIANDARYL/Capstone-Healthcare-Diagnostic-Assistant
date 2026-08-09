# ============================================================
# EVALUATION: Metrics
# Measures performance of the AI diagnostic models
# ============================================================

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


class DiagnosticEvaluator:
    """
    Evaluation engine for the AI diagnostic system.

    Calculates:
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - Confusion Matrix
        - ROC-AUC
    """

    def __init__(self, class_labels=None):
        self.class_labels = class_labels or [
            'flu',
            'covid19',
            'dengue',
            'cardiac_event',
            'diabetes',
            'common_cold',
            'tuberculosis',
            'meningitis'
        ]

        self.results = {}

    # --------------------------------------------------------
    # BASIC CLASSIFICATION METRICS
    # --------------------------------------------------------

    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate all required evaluation metrics.
        """

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=self.class_labels
        )

        # ROC-AUC requires probability predictions.
        roc_auc = None

        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class='ovr',
                    average='weighted',
                    labels=self.class_labels
                )
            except ValueError:
                # Happens if a test set does not contain
                # every possible class.
                roc_auc = None

        return {
            'accuracy': round(float(accuracy), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': (
                round(float(roc_auc), 4)
                if roc_auc is not None else None
            ),
            'confusion_matrix': cm
        }

    # --------------------------------------------------------
    # STORE MODEL RESULTS
    # --------------------------------------------------------

    def evaluate_model(
        self,
        model_name,
        y_true,
        y_pred,
        y_prob=None
    ):
        """
        Evaluate one diagnostic model and store its results.
        """

        metrics = self.calculate_metrics(
            y_true,
            y_pred,
            y_prob
        )

        self.results[model_name] = metrics

        return metrics

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    def get_classification_report(self, y_true, y_pred):
        """
        Generate detailed per-disease classification report.
        """

        return classification_report(
            y_true,
            y_pred,
            labels=self.class_labels,
            target_names=self.class_labels,
            zero_division=0
        )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    def get_comparison_dataframe(self):
        """
        Convert stored model results into a pandas DataFrame.
        """

        rows = []

        for model_name, metrics in self.results.items():

            rows.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    def print_results(self):
        """
        Display evaluation results in a readable format.
        """

        print("\n")
        print("=" * 70)
        print("AI DIAGNOSTIC SYSTEM — EVALUATION RESULTS")
        print("=" * 70)

        for model_name, metrics in self.results.items():

            print(f"\n{model_name}")
            print("-" * 50)

            print(
                f"Accuracy  : {metrics['accuracy']:.2%}"
            )

            print(
                f"Precision : {metrics['precision']:.2%}"
            )

            print(
                f"Recall    : {metrics['recall']:.2%}"
            )

            print(
                f"F1-Score  : {metrics['f1_score']:.2%}"
            )

            if metrics['roc_auc'] is not None:
                print(
                    f"ROC-AUC   : {metrics['roc_auc']:.2%}"
                )
            else:
                print("ROC-AUC   : N/A")

        print("\n" + "=" * 70)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    def save_results(self, filename='evaluation_results.csv'):
        """
        Save evaluation metrics to CSV.
        """

        df = self.get_comparison_dataframe()

        df.to_csv(
            filename,
            index=False
        )

        print(f"Evaluation results saved to: {filename}")

        return df