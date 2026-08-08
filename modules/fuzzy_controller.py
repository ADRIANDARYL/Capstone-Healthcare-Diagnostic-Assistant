# ============================================================
# MODULE 6: Fuzzy Logic — Patient Severity Assessment
# Covers: Week 12 (Fuzzy Logic)
# ============================================================

from typing import Dict
import numpy as np


class FuzzySeverityAssessor:
    """
    Fuzzy Logic system for patient severity assessment.

    Inputs:
        - Temperature (°C)
        - Heart Rate (beats per minute)
        - Number of Symptoms

    Output:
        - Severity Score (0–100)
        - Severity Label

    Fuzzy Logic Pipeline:
        1. Fuzzification
        2. Rule Evaluation
        3. Defuzzification
        4. Severity Classification
    """

    # ========================================================
    # TEMPERATURE MEMBERSHIP FUNCTIONS
    # ========================================================

    def _membership_temp(self, temp: float) -> Dict[str, float]:
        """
        Convert temperature into fuzzy membership degrees.

        Fuzzy sets:
            normal
            mild
            high
            critical
        """

        normal = (
            (37.5 - temp) / 1.0
            if temp <= 37.5
            else 0
        )

        mild = max(
            0,
            1 - abs(temp - 38.0) / 1.0
        )

        high = max(
            0,
            1 - abs(temp - 39.0) / 1.0
        )

        critical = (
            (temp - 39.0) / 1.5
            if temp >= 39.0
            else 0
        )

        return {
            'normal': self._clamp(normal),
            'mild': self._clamp(mild),
            'high': self._clamp(high),
            'critical': self._clamp(critical)
        }

    # ========================================================
    # HEART RATE MEMBERSHIP FUNCTIONS
    # ========================================================

    def _membership_hr(self, hr: int) -> Dict[str, float]:
        """
        Convert heart rate into fuzzy membership degrees.

        Fuzzy sets:
            low
            normal
            elevated
            high
        """

        low = (
            (70 - hr) / 10.0
            if hr <= 70
            else 0
        )

        normal = max(
            0,
            1 - abs(hr - 80) / 20.0
        )

        elevated = max(
            0,
            1 - abs(hr - 100) / 15.0
        )

        high = (
            (hr - 100) / 20.0
            if hr >= 100
            else 0
        )

        return {
            'low': self._clamp(low),
            'normal': self._clamp(normal),
            'elevated': self._clamp(elevated),
            'high': self._clamp(high)
        }

    # ========================================================
    # SYMPTOM COUNT MEMBERSHIP FUNCTIONS
    # ========================================================

    def _membership_symptoms(self, count: int) -> Dict[str, float]:
        """
        Convert the number of symptoms into fuzzy membership degrees.

        Fuzzy sets:
            few
            moderate
            many
        """

        few = max(
            0,
            min(1, (3 - count) / 2.0)
        )

        moderate = max(
            0,
            1 - abs(count - 4) / 2.0
        )

        many = max(
            0,
            min(1, (count - 5) / 3.0)
        )

        return {
            'few': self._clamp(few),
            'moderate': self._clamp(moderate),
            'many': self._clamp(many)
        }

    # ========================================================
    # FUZZY RULE EVALUATION
    # ========================================================

    def _evaluate_rules(
        self,
        temp_mf: Dict[str, float],
        hr_mf: Dict[str, float],
        symptom_mf: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evaluate fuzzy IF-THEN rules.

        AND  → min()
        OR   → max()
        """

        rules = {

            # ------------------------------------------------
            # CRITICAL RULES
            # ------------------------------------------------
            'critical': max(
                min(
                    temp_mf['critical'],
                    hr_mf['high']
                ),

                min(
                    temp_mf['critical'],
                    symptom_mf['many']
                )
            ),

            # ------------------------------------------------
            # HIGH RULES
            # ------------------------------------------------
            'high': max(
                min(
                    temp_mf['high'],
                    hr_mf['elevated']
                ),

                min(
                    temp_mf['high'],
                    symptom_mf['many']
                ),

                min(
                    temp_mf['mild'],
                    hr_mf['high']
                )
            ),

            # ------------------------------------------------
            # MODERATE RULES
            # ------------------------------------------------
            'moderate': max(
                min(
                    temp_mf['mild'],
                    hr_mf['normal']
                ),

                min(
                    temp_mf['high'],
                    symptom_mf['moderate']
                ),

                min(
                    temp_mf['normal'],
                    symptom_mf['many']
                )
            ),

            # ------------------------------------------------
            # MILD RULES
            # ------------------------------------------------
            'mild': max(
                min(
                    temp_mf['mild'],
                    symptom_mf['few']
                ),

                min(
                    temp_mf['normal'],
                    symptom_mf['moderate']
                )
            ),

            # ------------------------------------------------
            # LOW RULE
            # ------------------------------------------------
            'low': min(
                temp_mf['normal'],
                hr_mf['normal'],
                symptom_mf['few']
            )
        }

        return {
            key: self._clamp(value)
            for key, value in rules.items()
        }

    # ========================================================
    # DEFUZZIFICATION
    # ========================================================

    def _defuzzify(
        self,
        severity_rules: Dict[str, float]
    ) -> float:
        """
        Convert fuzzy severity values into a crisp
        severity score from 0–100.

        Centroid-style weighted calculation.
        """

        centers = {
            'low': 15,
            'mild': 35,
            'moderate': 55,
            'high': 75,
            'critical': 92
        }

        numerator = sum(
            centers[label] * strength
            for label, strength in severity_rules.items()
            if label in centers
        )

        denominator = sum(
            severity_rules.values()
        )

        # Prevent division by zero
        if denominator <= 0:
            return 0.0

        score = numerator / denominator

        return max(0.0, min(100.0, score))

    # ========================================================
    # SEVERITY CLASSIFICATION
    # ========================================================

    def _classify(self, score: float) -> str:
        """
        Convert numerical severity score into a label.

        0–20    → LOW
        20–40   → MILD
        40–60   → MODERATE
        60–80   → HIGH
        80–100  → CRITICAL
        """

        if score >= 80:
            return "CRITICAL"

        elif score >= 60:
            return "HIGH"

        elif score >= 40:
            return "MODERATE"

        elif score >= 20:
            return "MILD"

        return "LOW"

    # ========================================================
    # MAIN ASSESSMENT FUNCTION
    # ========================================================

    def assess(
        self,
        temperature: float,
        heart_rate: int,
        symptom_count: int
    ) -> Dict:
        """
        Complete fuzzy inference pipeline.

        Step 1: Fuzzification
        Step 2: Rule Evaluation
        Step 3: Defuzzification
        Step 4: Classification
        """

        # ----------------------------------------------------
        # STEP 1: FUZZIFICATION
        # ----------------------------------------------------

        temp_mf = self._membership_temp(temperature)

        hr_mf = self._membership_hr(heart_rate)

        symptom_mf = self._membership_symptoms(
            symptom_count
        )

        # ----------------------------------------------------
        # STEP 2: RULE EVALUATION
        # ----------------------------------------------------

        rules = self._evaluate_rules(
            temp_mf,
            hr_mf,
            symptom_mf
        )

        # ----------------------------------------------------
        # STEP 3: DEFUZZIFICATION
        # ----------------------------------------------------

        severity_score = self._defuzzify(rules)

        # ----------------------------------------------------
        # STEP 4: CLASSIFICATION
        # ----------------------------------------------------

        severity_label = self._classify(
            severity_score
        )

        return {
            'severity_score': round(
                severity_score,
                2
            ),

            'severity_label': severity_label,

            'rule_strengths': {
                key: round(value, 3)
                for key, value in rules.items()
            },

            'memberships': {
                'temperature': temp_mf,
                'heart_rate': hr_mf,
                'symptoms': symptom_mf
            }
        }

    # ========================================================
    # AGENT INTERFACE
    # ========================================================

    def analyze(self, percept) -> Dict:
        """
        Standard interface used by Module 1.

        Module 1 provides:
            percept.temperature
            percept.heart_rate
            percept.symptoms
        """

        result = self.assess(
            percept.temperature,
            percept.heart_rate,
            len(percept.symptoms)
        )

        result['summary'] = (
            f"Severity: "
            f"{result['severity_label']} "
            f"({result['severity_score']:.1f}/100)"
        )

        # Allows Module 1 to treat this as an analysis module
        result['diagnosis'] = result['severity_label']

        # Severity score acts as the module's confidence value
        result['confidence'] = (
            result['severity_score'] / 100
        )

        return result

    # ========================================================
    # UTILITY
    # ========================================================

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0
    ) -> float:
        """
        Ensure membership values stay between 0 and 1.
        """

        return max(
            minimum,
            min(maximum, float(value))
        )


# ============================================================
# MODULE 6 TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODULE 6 — FUZZY LOGIC SEVERITY ASSESSMENT TEST")
    print("=" * 60)

    assessor = FuzzySeverityAssessor()

    test_cases = [
        (37.0, 72, 2, "Normal patient"),
        (38.5, 95, 4, "Mild illness"),
        (39.8, 115, 7, "Severe case"),
        (40.2, 130, 9, "Critical case")
    ]

    for temperature, heart_rate, symptom_count, description in test_cases:

        result = assessor.assess(
            temperature,
            heart_rate,
            symptom_count
        )

        print(f"\n{description}")
        print("-" * 40)

        print(
            f"Temperature : {temperature:.1f} °C"
        )

        print(
            f"Heart Rate  : {heart_rate} bpm"
        )

        print(
            f"Symptoms    : {symptom_count}"
        )

        print(
            f"Severity    : "
            f"{result['severity_score']}/100"
        )

        print(
            f"Label       : "
            f"{result['severity_label']}"
        )

        print("\nMemberships:")

        print(
            "  Temperature:",
            result['memberships']['temperature']
        )

        print(
            "  Heart Rate :",
            result['memberships']['heart_rate']
        )

        print(
            "  Symptoms   :",
            result['memberships']['symptoms']
        )

    print("\n" + "=" * 60)
    print("Module 6 test completed.")
    print("=" * 60)