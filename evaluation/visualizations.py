# ============================================================
# EVALUATION: Visualizations
# Generates confusion matrices and model comparison charts
# ============================================================

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluationVisualizer:
    """
    Generates visualizations for AI diagnostic evaluation.
    """

    def __init__(self, output_dir='evaluation/results'):

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    def plot_confusion_matrix(
        self,
        confusion_matrix,
        labels,
        model_name
    ):
        """
        Generate and save a confusion matrix.
        """

        plt.figure(
            figsize=(10, 8)
        )

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            xticklabels=labels,
            yticklabels=labels
        )

        plt.title(
            f'Confusion Matrix — {model_name}'
        )

        plt.xlabel(
            'Predicted Diagnosis'
        )

        plt.ylabel(
            'Actual Diagnosis'
        )

        plt.xticks(
            rotation=45,
            ha='right'
        )

        plt.tight_layout()

        filename = os.path.join(
            self.output_dir,
            f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        )

        plt.savefig(
            filename,
            dpi=150,
            bbox_inches='tight'
        )

        plt.show()

        print(
            f"Saved confusion matrix: {filename}"
        )

        return filename

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    def plot_model_comparison(
        self,
        results
    ):
        """
        Generate comparison bar chart for all models.
        """

        metrics = [
            'Accuracy',
            'Precision',
            'Recall',
            'F1-Score',
            'ROC-AUC'
        ]

        models = results['Model'].tolist()

        x = np.arange(
            len(models)
        )

        width = 0.15

        plt.figure(
            figsize=(14, 7)
        )

        for i, metric in enumerate(metrics):

            if metric not in results.columns:
                continue

            values = results[metric].fillna(0).values

            plt.bar(
                x + i * width,
                values,
                width,
                label=metric
            )

        plt.xlabel(
            'AI Model'
        )

        plt.ylabel(
            'Score'
        )

        plt.title(
            'AI Diagnostic Model Performance Comparison'
        )

        plt.xticks(
            x + width * 2,
            models,
            rotation=20
        )

        plt.ylim(
            0,
            1.05
        )

        plt.legend()

        plt.grid(
            axis='y',
            alpha=0.3
        )

        plt.tight_layout()

        filename = os.path.join(
            self.output_dir,
            'model_comparison.png'
        )

        plt.savefig(
            filename,
            dpi=150,
            bbox_inches='tight'
        )

        plt.show()

        print(
            f"Saved model comparison: {filename}"
        )

        return filename

    # --------------------------------------------------------
    # ALL CONFUSION MATRICES
    # --------------------------------------------------------

    def plot_all_confusion_matrices(
        self,
        evaluation_results,
        labels
    ):
        """
        Generate confusion matrices for every evaluated model.
        """

        files = []

        for model_name, metrics in evaluation_results.items():

            if 'confusion_matrix' not in metrics:
                continue

            filename = self.plot_confusion_matrix(
                metrics['confusion_matrix'],
                labels,
                model_name
            )

            files.append(filename)

        return files