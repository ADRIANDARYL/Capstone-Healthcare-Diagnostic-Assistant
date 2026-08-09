# ============================================================
# MODULE 3: Bayesian Network — Probabilistic Diagnosis
# Covers: Week 7 (Bayesian Networks & Probabilistic Reasoning)
# ============================================================

import numpy as np
from typing import Dict, List


class SimpleBayesianDiagnostics:
    """
    Simplified Bayesian diagnostic model using
    pre-computed conditional probabilities.

    Uses Naïve Bayes:
        P(Disease | Symptoms)
        ∝ P(Disease) × ∏ P(Symptom | Disease)

    Log probabilities are used to avoid numerical
    underflow when multiplying many small probabilities.
    """

    def __init__(self):

        # ----------------------------------------------------
        # STEP 1: Prior probabilities P(Disease)
        # ----------------------------------------------------
        self.priors = {
            'flu':         0.15,
            'covid19':     0.08,
            'dengue':      0.05,
            'cardiac':     0.04,
            'diabetes':    0.10,
            'common_cold': 0.30,
            'healthy':     0.28,
        }

        # ----------------------------------------------------
        # STEP 2: Likelihood table
        # P(Symptom | Disease)
        # ----------------------------------------------------
        self.likelihoods = {

            'flu': {
                'fever': 0.90,
                'cough': 0.85,
                'fatigue': 0.88,
                'headache': 0.70,
                'body_aches': 0.80,
                'loss_of_smell': 0.20,
                'chest_pain': 0.05,
                'rash': 0.05,
                'joint_pain': 0.40,
            },

            'covid19': {
                'fever': 0.88,
                'cough': 0.80,
                'fatigue': 0.90,
                'loss_of_smell': 0.85,
                'headache': 0.65,
                'body_aches': 0.60,
                'chest_pain': 0.20,
                'rash': 0.05,
                'joint_pain': 0.20,
            },

            'dengue': {
                'fever': 0.98,
                'rash': 0.75,
                'joint_pain': 0.85,
                'headache': 0.90,
                'fatigue': 0.80,
                'cough': 0.15,
                'loss_of_smell': 0.05,
                'chest_pain': 0.05,
                'body_aches': 0.88,
            },

            'cardiac': {
                'chest_pain': 0.92,
                'shortness_of_breath': 0.88,
                'fatigue': 0.70,
                'sweating': 0.75,
                'fever': 0.10,
                'cough': 0.15,
                'rash': 0.02,
                'joint_pain': 0.10,
                'headache': 0.30,
            },

            'diabetes': {
                'fatigue': 0.82,
                'frequent_urination': 0.95,
                'excessive_thirst': 0.92,
                'blurred_vision': 0.70,
                'fever': 0.10,
                'cough': 0.05,
                'rash': 0.08,
                'headache': 0.40,
                'joint_pain': 0.20,
            },

            'common_cold': {
                'cough': 0.90,
                'fever': 0.50,
                'headache': 0.60,
                'fatigue': 0.55,
                'body_aches': 0.50,
                'loss_of_smell': 0.30,
                'rash': 0.02,
                'chest_pain': 0.05,
                'joint_pain': 0.15,
            },

            'healthy': {
                'fever': 0.02,
                'cough': 0.05,
                'fatigue': 0.10,
                'headache': 0.08,
                'rash': 0.01,
                'chest_pain': 0.01,
                'joint_pain': 0.05,
                'loss_of_smell': 0.01,
                'body_aches': 0.05,
            }
        }

    # --------------------------------------------------------
    # STEP 3: Compute posterior probabilities
    # --------------------------------------------------------

    def compute_posterior(
        self,
        symptoms: List[str]
    ) -> Dict[str, float]:
        """
        Calculate posterior probabilities using Naïve Bayes.

        P(D|S1,...,Sn)
        ∝ P(D) × P(S1|D) × ... × P(Sn|D)

        Logarithms are used to prevent numerical underflow.
        """

        # Clean symptom names
        symptoms_clean = [
            symptom.strip().lower().replace(' ', '_')
            for symptom in symptoms
        ]

        log_scores = {}

        # Calculate log probability for each disease
        for disease, prior in self.priors.items():

            log_probability = np.log(prior)

            for symptom in symptoms_clean:

                # 0.01 prevents log(0)
                likelihood = self.likelihoods[disease].get(
                    symptom,
                    0.01
                )

                log_probability += np.log(likelihood)

            log_scores[disease] = log_probability

        # ----------------------------------------------------
        # Convert log scores back to probabilities
        # using a numerically stable softmax-style calculation
        # ----------------------------------------------------

        max_log = max(log_scores.values())

        exp_scores = {
            disease: np.exp(score - max_log)
            for disease, score in log_scores.items()
        }

        total = sum(exp_scores.values())

        posteriors = {
            disease: round(score / total, 4)
            for disease, score in exp_scores.items()
        }

        return posteriors

    # --------------------------------------------------------
    # STEP 4: Agent interface
    # --------------------------------------------------------

    def analyze(self, percept) -> Dict:
        """
        Standard interface called by Module 1.

        Receives a PatientPercept and returns:
        - top diagnosis
        - confidence
        - all posterior probabilities
        - ranked diagnoses
        """

        posteriors = self.compute_posterior(
            percept.symptoms
        )

        # Highest probability diagnosis
        top_disease = max(
            posteriors,
            key=posteriors.get
        )

        top_probability = posteriors[top_disease]

        # Rank all diseases
        ranked_diagnoses = sorted(
            posteriors.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return {
            'summary': (
                f"Top diagnosis: {top_disease} "
                f"({top_probability:.2%})"
            ),

            'diagnosis': top_disease,

            'confidence': top_probability,

            'all_posteriors': posteriors,

            'ranked_diagnoses': ranked_diagnoses[:5]
        }

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    def explain(
        self,
        disease: str,
        symptoms: List[str]
    ) -> str:
        """
        Explain the probability calculation for
        a particular disease.
        """

        disease = disease.lower().replace(' ', '_')

        if disease not in self.priors:
            return f"Unknown disease: {disease}"

        symptoms_clean = [
            symptom.strip().lower().replace(' ', '_')
            for symptom in symptoms
        ]

        evidence = []

        for symptom in symptoms_clean:

            likelihood = self.likelihoods[
                disease
            ].get(symptom, 0.01)

            evidence.append(
                f"P({symptom}|{disease})={likelihood:.2f}"
            )

        return (
            f"P({disease}) = {self.priors[disease]:.2f} × "
            + " × ".join(evidence)
        )


# ============================================================
# MODULE 3 TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODULE 3 — BAYESIAN DIAGNOSTICS TEST")
    print("=" * 60)

    bn = SimpleBayesianDiagnostics()

    # Test symptoms from the PDF
    symptoms = [
        "fever",
        "cough",
        "loss of smell",
        "fatigue"
    ]

    print("\nPatient symptoms:")
    print(", ".join(symptoms))

    # Calculate posterior probabilities
    posteriors = bn.compute_posterior(symptoms)

    print("\nTop 3 Diagnoses")
    print("-" * 40)

    ranked = sorted(
        posteriors.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for disease, probability in ranked[:3]:
        print(
            f"{disease:<20}: "
            f"{probability:.2%}"
        )

    # Test explanation
    print("\nExplanation")
    print("-" * 40)

    print(
        bn.explain(
            "covid19",
            symptoms
        )
    )

    print("\nModule 3 test completed.")