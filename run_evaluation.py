# ============================================================
# RUN EVALUATION
# Evaluates Module 4 and Module 5
# Generates metrics, confusion matrices and comparison chart
# ============================================================

import os

from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel

from evaluation.metrics import DiagnosticEvaluator
from evaluation.visualizations import EvaluationVisualizer


# ============================================================
# TEST PATIENTS
# ============================================================

TEST_PATIENTS = [
    {
        "id": "P001",
        "diagnosis": "covid19",
        "symptoms": [
            "fever",
            "cough",
            "fatigue",
            "loss_of_smell"
        ]
    },
    {
        "id": "P002",
        "diagnosis": "dengue",
        "symptoms": [
            "fever",
            "rash",
            "joint_pain",
            "headache",
            "fatigue"
        ]
    },
    {
        "id": "P003",
        "diagnosis": "diabetes",
        "symptoms": [
            "fatigue",
            "frequent_urination",
            "excessive_thirst",
            "blurred_vision",
            "weight_loss"
        ]
    },
    {
        "id": "P004",
        "diagnosis": "cardiac_event",
        "symptoms": [
            "chest_pain",
            "shortness_of_breath",
            "sweating",
            "fatigue"
        ]
    },
    {
        "id": "P005",
        "diagnosis": "tuberculosis",
        "symptoms": [
            "cough",
            "fatigue",
            "weight_loss",
            "night_sweats",
            "fever"
        ]
    },
    {
        "id": "P006",
        "diagnosis": "flu",
        "symptoms": [
            "fever",
            "cough",
            "headache",
            "body_aches",
            "fatigue"
        ]
    },
    {
        "id": "P007",
        "diagnosis": "common_cold",
        "symptoms": [
            "cough",
            "runny_nose",
            "sneezing",
            "sore_throat"
        ]
    },
    {
        "id": "P008",
        "diagnosis": "meningitis",
        "symptoms": [
            "fever",
            "severe_headache",
            "stiff_neck",
            "vomiting",
            "confusion"
        ]
    }
]


# ============================================================
# DISEASE LABELS
# ============================================================

CLASS_LABELS = [
    "flu",
    "covid19",
    "dengue",
    "cardiac_event",
    "diabetes",
    "common_cold",
    "tuberculosis",
    "meningitis"
]


# ============================================================
# EXTRACT DIAGNOSIS
# ============================================================

def extract_diagnosis(result):
    """
    Extract the predicted diagnosis from a module's
    predict() result.

    Handles the dictionary format used by the modules.
    """

    if isinstance(result, dict):

        for key in [
            "diagnosis",
            "predicted_class",
            "prediction",
            "class"
        ]:

            if key in result:
                return str(result[key]).lower()

    return str(result).lower()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI DIAGNOSTIC SYSTEM — MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Create evaluation objects
    # --------------------------------------------------------

    evaluator = DiagnosticEvaluator(
        class_labels=CLASS_LABELS
    )

    visualizer = EvaluationVisualizer(
        output_dir="evaluation/results"
    )

    # --------------------------------------------------------
    # MODULE 4 — MACHINE LEARNING
    # --------------------------------------------------------

    print("\n")
    print("MODULE 4 — MACHINE LEARNING CLASSIFIER")
    print("-" * 70)

    ml = MLDiagnosticClassifier()

    ml_results = ml.train(verbose=True)

    print("\nModule 4 training completed.")

    # --------------------------------------------------------
    # MODULE 5 — DEEP NEURAL NETWORK
    # --------------------------------------------------------

    print("\n")
    print("MODULE 5 — DEEP NEURAL NETWORK")
    print("-" * 70)

    nn = NeuralDiagnosticModel()

    nn_results = nn.train(
        epochs=30,
        verbose=1
    )

    print("\nModule 5 training completed.")

    # --------------------------------------------------------
    # EVALUATE SAME PATIENTS WITH BOTH MODELS
    # --------------------------------------------------------

    y_true = []

    ml_predictions = []
    nn_predictions = []

    print("\n")
    print("=" * 70)
    print("TESTING MODELS ON COMMON TEST PATIENTS")
    print("=" * 70)

    for patient in TEST_PATIENTS:

        actual = patient["diagnosis"]

        y_true.append(actual)

        print(
            f"\n{patient['id']} "
            f"| Actual diagnosis: {actual}"
        )

        # ----------------------------------------------------
        # Module 4 prediction
        # ----------------------------------------------------

        try:

            ml_result = ml.predict(
                patient["symptoms"]
            )

            ml_prediction = extract_diagnosis(
                ml_result
            )

        except Exception as error:

            print(
                f"  ML prediction error: {error}"
            )

            ml_prediction = "unknown"

        ml_predictions.append(
            ml_prediction
        )

        print(
            f"  Module 4 prediction : "
            f"{ml_prediction}"
        )

        # ----------------------------------------------------
        # Module 5 prediction
        # ----------------------------------------------------

        try:

            nn_result = nn.predict(
                patient["symptoms"]
            )

            nn_prediction = extract_diagnosis(
                nn_result
            )

        except Exception as error:

            print(
                f"  Neural prediction error: {error}"
            )

            nn_prediction = "unknown"

        nn_predictions.append(
            nn_prediction
        )

        print(
            f"  Module 5 prediction : "
            f"{nn_prediction}"
        )

    # --------------------------------------------------------
    # Remove invalid predictions before metric calculation
    # --------------------------------------------------------

    valid_ml = [
        i for i, prediction
        in enumerate(ml_predictions)
        if prediction in CLASS_LABELS
    ]

    valid_nn = [
        i for i, prediction
        in enumerate(nn_predictions)
        if prediction in CLASS_LABELS
    ]

    # --------------------------------------------------------
    # MODULE 4 METRICS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODULE 4 — EVALUATION METRICS")
    print("=" * 70)

    if valid_ml:

        ml_y_true = [
            y_true[i]
            for i in valid_ml
        ]

        ml_y_pred = [
            ml_predictions[i]
            for i in valid_ml
        ]

        ml_metrics = evaluator.evaluate_model(
            "Module 4 - ML Classifier",
            ml_y_true,
            ml_y_pred
        )

        print(
            f"Accuracy  : "
            f"{ml_metrics['accuracy']:.2%}"
        )

        print(
            f"Precision : "
            f"{ml_metrics['precision']:.2%}"
        )

        print(
            f"Recall    : "
            f"{ml_metrics['recall']:.2%}"
        )

        print(
            f"F1-Score  : "
            f"{ml_metrics['f1_score']:.2%}"
        )

    else:

        print("No valid ML predictions available.")

    # --------------------------------------------------------
    # MODULE 5 METRICS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODULE 5 — EVALUATION METRICS")
    print("=" * 70)

    if valid_nn:

        nn_y_true = [
            y_true[i]
            for i in valid_nn
        ]

        nn_y_pred = [
            nn_predictions[i]
            for i in valid_nn
        ]

        nn_metrics = evaluator.evaluate_model(
            "Module 5 - Neural Network",
            nn_y_true,
            nn_y_pred
        )

        print(
            f"Accuracy  : "
            f"{nn_metrics['accuracy']:.2%}"
        )

        print(
            f"Precision : "
            f"{nn_metrics['precision']:.2%}"
        )

        print(
            f"Recall    : "
            f"{nn_metrics['recall']:.2%}"
        )

        print(
            f"F1-Score  : "
            f"{nn_metrics['f1_score']:.2%}"
        )

    else:

        print("No valid Neural Network predictions available.")

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GENERATING CONFUSION MATRICES")
    print("=" * 70)

    evaluation_results = evaluator.results

    confusion_files = visualizer.plot_all_confusion_matrices(
        evaluation_results,
        CLASS_LABELS
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GENERATING MODEL COMPARISON")
    print("=" * 70)

    comparison_df = evaluator.get_comparison_dataframe()

    if not comparison_df.empty:

        print("\n")
        print(comparison_df.to_string(index=False))

        comparison_file = visualizer.plot_model_comparison(
            comparison_df
        )

        print(
            f"\nModel comparison saved to: "
            f"{comparison_file}"
        )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    results_file = os.path.join(
        "evaluation",
        "results",
        "evaluation_results.csv"
    )

    evaluator.save_results(
        results_file
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "\nTest patients evaluated:",
        len(TEST_PATIENTS)
    )

    print(
        "\nGenerated files:"
    )

    for file in confusion_files:
        print(
            f"  ✓ {file}"
        )

    if not comparison_df.empty:
        print(
            f"  ✓ {comparison_file}"
        )

    print(
        f"  ✓ {results_file}"
    )

    print("\n")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()