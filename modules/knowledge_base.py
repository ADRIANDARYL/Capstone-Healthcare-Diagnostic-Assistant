# ============================================================
# MODULE 2: FOL Knowledge Base + Inference Engine
# Covers: Week 5 (First-Order Logic & Inference)
# ============================================================

from typing import Set, List, Dict, Tuple, Optional


class MedicalKnowledgeBase:
    """
    First-Order Logic based medical knowledge base.

    Supports:
    - Knowledge representation using facts and rules
    - Forward chaining
    - Backward chaining
    - Certainty-weighted inference
    - Patient symptom and vital processing

    This module provides educational/project-level
    rule-based inference and is not a replacement
    for professional medical diagnosis.
    """

    def __init__(self):

        # Known facts for the current patient
        #
        # Example:
        # {
        #     "fever": 1.0,
        #     "cough": 1.0
        # }
        self.facts: Set[str] = set()

        # Rules are stored as:
        #
        # (
        #     ["fever", "cough"],
        #     "flu_suspected",
        #     0.75
        # )
        self.rules: List[Tuple[List[str], str, float]] = []

        # Certainty factor associated with each fact
        self.certainty_factors: Dict[str, float] = {}

        # Store inference steps for explanations/debugging
        self.inference_history: List[str] = []

        # Load the permanent medical knowledge
        self._load_medical_knowledge()

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_fact(fact: str) -> str:
        """
        Convert a fact/symptom to lower_snake_case.

        Examples:
            'Loss of Smell' -> 'loss_of_smell'
            'High Fever'   -> 'high_fever'
            'Cough'        -> 'cough'
        """

        if not isinstance(fact, str):
            raise TypeError("Fact must be a string.")

        fact = fact.strip().lower()

        # Convert spaces and hyphens to underscores
        fact = fact.replace(" ", "_")
        fact = fact.replace("-", "_")

        # Prevent duplicate underscores
        while "__" in fact:
            fact = fact.replace("__", "_")

        return fact.strip("_")

    # ========================================================
    # LOAD MEDICAL KNOWLEDGE
    # ========================================================

    def _load_medical_knowledge(self):
        """
        Load domain medical knowledge as logical rules.

        Rule format:

        (
            [conditions],
            conclusion,
            certainty_factor
        )
        """

        disease_rules = [

            # ------------------------------------------------
            # FLU
            # ------------------------------------------------

            (
                ["fever", "cough", "fatigue"],
                "flu_suspected",
                0.75
            ),

            (
                ["fever", "headache", "body_aches"],
                "flu_possible",
                0.80
            ),

            (
                ["flu_suspected", "high_fever"],
                "flu_confirmed",
                0.85
            ),

            # ------------------------------------------------
            # COVID-19
            # ------------------------------------------------

            (
                [
                    "fever",
                    "cough",
                    "loss_of_smell",
                    "fatigue"
                ],
                "covid19_suspected",
                0.85
            ),

            # PDF example:
            # COVID-19 suspected + fatigue
            # -> COVID-19 likely

            (
                [
                    "covid19_suspected",
                    "fatigue"
                ],
                "covid19_likely",
                0.90
            ),

            (
                [
                    "covid19_suspected",
                    "positive_pcr"
                ],
                "covid19_confirmed",
                0.99
            ),

            # ------------------------------------------------
            # DENGUE
            # ------------------------------------------------

            (
                ["fever", "rash", "joint_pain"],
                "dengue_suspected",
                0.80
            ),

            # ------------------------------------------------
            # CARDIAC CONDITIONS
            # ------------------------------------------------

            (
                [
                    "chest_pain",
                    "shortness_of_breath",
                    "sweating"
                ],
                "cardiac_event_suspected",
                0.90
            ),

            (
                [
                    "cardiac_event_suspected",
                    "elevated_troponin"
                ],
                "myocardial_infarction",
                0.95
            ),

            # ------------------------------------------------
            # MENINGITIS
            # ------------------------------------------------

            (
                [
                    "headache",
                    "stiff_neck",
                    "high_fever",
                    "light_sensitivity"
                ],
                "meningitis_suspected",
                0.88
            ),

            # ------------------------------------------------
            # TUBERCULOSIS
            # ------------------------------------------------

            (
                [
                    "cough",
                    "weight_loss",
                    "night_sweats",
                    "fatigue"
                ],
                "tuberculosis_suspected",
                0.82
            ),

            # ------------------------------------------------
            # DIABETES
            # ------------------------------------------------

            (
                [
                    "frequent_urination",
                    "excessive_thirst",
                    "blurred_vision"
                ],
                "diabetes_suspected",
                0.78
            ),

            # ------------------------------------------------
            # URGENCY / ACTION RULES
            # ------------------------------------------------

            (
                ["myocardial_infarction"],
                "EMERGENCY",
                1.00
            ),

            (
                ["meningitis_suspected"],
                "EMERGENCY",
                0.95
            ),

            (
                ["covid19_confirmed"],
                "ISOLATE_AND_TREAT",
                0.99
            ),

            (
                ["flu_confirmed"],
                "REST_AND_MEDICATE",
                0.90
            ),
        ]

        for conditions, conclusion, certainty in disease_rules:
            self.add_rule(
                conditions,
                conclusion,
                certainty
            )

    # ========================================================
    # ADD FACT
    # ========================================================

    def add_fact(
        self,
        fact: str,
        certainty: float = 1.0
    ):
        """
        Add a fact to the knowledge base.

        If a fact already exists, the strongest certainty
        factor is retained.
        """

        fact = self.normalize_fact(fact)

        certainty = max(
            0.0,
            min(1.0, float(certainty))
        )

        self.facts.add(fact)

        existing = self.certainty_factors.get(
            fact,
            0.0
        )

        self.certainty_factors[fact] = max(
            existing,
            certainty
        )

    # ========================================================
    # ADD RULE
    # ========================================================

    def add_rule(
        self,
        conditions: List[str],
        conclusion: str,
        certainty: float = 1.0
    ):
        """
        Add a logical rule to the knowledge base.

        Example:

            fever AND cough
            ->
            flu_suspected

        CF = 0.75
        """

        if not conditions:
            raise ValueError(
                "A rule must contain at least one condition."
            )

        normalized_conditions = [
            self.normalize_fact(condition)
            for condition in conditions
        ]

        normalized_conclusion = (
            self.normalize_fact(conclusion)
        )

        certainty = max(
            0.0,
            min(1.0, float(certainty))
        )

        self.rules.append(
            (
                normalized_conditions,
                normalized_conclusion,
                certainty
            )
        )

    # ========================================================
    # CLEAR CURRENT PATIENT DATA
    # ========================================================

    def clear_facts(self):
        """
        Clear facts and inference history belonging
        to the previous patient.

        The medical rules remain loaded.
        """

        self.facts.clear()
        self.certainty_factors.clear()
        self.inference_history.clear()

    # ========================================================
    # LOAD PATIENT SYMPTOMS
    # ========================================================

    def load_patient_symptoms(
        self,
        symptoms: List[str]
    ):
        """
        Convert patient symptoms into normalized facts.

        Example:

            'Loss of Smell'

        becomes:

            'loss_of_smell'
        """

        if not symptoms:
            return

        for symptom in symptoms:

            if not isinstance(symptom, str):
                continue

            normalized = self.normalize_fact(
                symptom
            )

            if normalized:
                self.add_fact(
                    normalized,
                    1.0
                )

    # ========================================================
    # LOAD PATIENT VITALS
    # ========================================================

    def load_patient_vitals(
        self,
        temperature: Optional[float] = None,
        heart_rate: Optional[int] = None
    ):
        """
        Convert important vital signs into logical facts.

        PDF requirement:

            temperature > 38°C
            -> fever

        Additional rules:
            temperature > 39.5°C -> high_fever
            heart rate > 100      -> tachycardia
        """

        if temperature is not None:

            if temperature > 38.0:

                # PDF requirement
                self.add_fact(
                    "fever",
                    1.0
                )

            if temperature > 39.5:

                self.add_fact(
                    "high_fever",
                    1.0
                )

        if heart_rate is not None:

            if heart_rate > 100:

                self.add_fact(
                    "tachycardia",
                    1.0
                )

    # ========================================================
    # FORWARD CHAINING
    # ========================================================

    def forward_chain(
        self,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Forward Chaining.

        Starts with known facts and repeatedly applies
        rules whose conditions are satisfied.

        Certainty calculation:

            Rule CF × minimum(condition CFs)

        Stops when no new/stronger facts can be inferred.
        """

        iteration = 0

        changed = True

        while changed:

            changed = False
            iteration += 1

            for conditions, conclusion, rule_cf in self.rules:

                # Check whether ALL conditions are known
                if not all(
                    condition in self.facts
                    for condition in conditions
                ):
                    continue

                # Get certainty of every condition
                condition_cfs = [
                    self.certainty_factors.get(
                        condition,
                        1.0
                    )
                    for condition in conditions
                ]

                # PDF requirement:
                #
                # Rule CF × minimum condition CF
                #
                combined_cf = (
                    rule_cf
                    * min(condition_cfs)
                )

                existing_cf = (
                    self.certainty_factors.get(
                        conclusion,
                        0.0
                    )
                )

                # Only add/update if this inference is
                # stronger than what we already know.
                if combined_cf > existing_cf:

                    self.facts.add(
                        conclusion
                    )

                    self.certainty_factors[
                        conclusion
                    ] = round(
                        combined_cf,
                        4
                    )

                    changed = True

                    cond_str = " ∧ ".join(
                        conditions
                    )

                    inference = (
                        f"Iter {iteration}: "
                        f"{cond_str} → "
                        f"{conclusion} "
                        f"(CF={combined_cf:.3f})"
                    )

                    self.inference_history.append(
                        inference
                    )

                    if verbose:
                        print(inference)

        return dict(
            self.certainty_factors
        )

    # ========================================================
    # BACKWARD CHAINING
    # ========================================================

    def backward_chain(
        self,
        goal: str,
        visited: Optional[Set[str]] = None,
        depth: int = 0
    ) -> Tuple[bool, float]:
        """
        Backward Chaining.

        Starts with a goal and works backwards to determine
        whether the required facts can prove the goal.

        Certainty calculation:

            Rule CF × minimum(condition CFs)

        A visited set prevents infinite recursion.
        """

        goal = self.normalize_fact(
            goal
        )

        if visited is None:
            visited = set()

        # Prevent infinite loops
        if goal in visited:
            return False, 0.0

        # ----------------------------------------------------
        # Goal already known
        # ----------------------------------------------------

        if goal in self.facts:

            return (
                True,
                self.certainty_factors.get(
                    goal,
                    1.0
                )
            )

        # Mark goal as visited
        visited = visited.copy()
        visited.add(goal)

        best_cf = 0.0

        # ----------------------------------------------------
        # Find rules that can conclude the goal
        # ----------------------------------------------------

        for conditions, conclusion, rule_cf in self.rules:

            if conclusion != goal:
                continue

            condition_results = []

            successful = True

            # Try to prove every condition
            for condition in conditions:

                proved, condition_cf = (
                    self.backward_chain(
                        condition,
                        visited.copy(),
                        depth + 1
                    )
                )

                if not proved:

                    successful = False
                    break

                condition_results.append(
                    condition_cf
                )

            if successful and condition_results:

                combined_cf = (
                    rule_cf
                    * min(condition_results)
                )

                best_cf = max(
                    best_cf,
                    combined_cf
                )

        if best_cf > 0:

            return (
                True,
                round(best_cf, 4)
            )

        return False, 0.0

    # ========================================================
    # GET TOP DIAGNOSIS
    # ========================================================

    def get_top_diagnosis(
        self,
        inferred: Optional[Dict[str, float]] = None
    ) -> Tuple[str, float]:
        """
        Return the diagnosis with the highest certainty.

        Diagnostic conclusions are identified using:
            *_suspected
            *_possible
            *_confirmed
            *_likely
        """

        source = (
            inferred
            if inferred is not None
            else self.certainty_factors
        )

        diagnoses = {
            fact: confidence
            for fact, confidence in source.items()
            if (
                fact.endswith("_suspected")
                or fact.endswith("_possible")
                or fact.endswith("_confirmed")
                or fact.endswith("_likely")
            )
        }

        if not diagnoses:

            return (
                "Unknown",
                0.0
            )

        diagnosis, confidence = max(
            diagnoses.items(),
            key=lambda item: item[1]
        )

        return (
            diagnosis,
            confidence
        )

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        percept
    ) -> Dict:
        """
        Standard interface called by Module 1.

        Required sequence from the PDF:

        1. Clear old facts
        2. Load patient symptoms
        3. Add vital signs
        4. Run forward chaining
        5. Return top diagnosis and confidence
        """

        # ----------------------------------------------------
        # 1. Clear old patient data
        # ----------------------------------------------------

        self.clear_facts()

        # ----------------------------------------------------
        # 2. Load symptoms
        # ----------------------------------------------------

        self.load_patient_symptoms(
            percept.symptoms
        )

        # ----------------------------------------------------
        # 3. Load vitals
        # ----------------------------------------------------

        self.load_patient_vitals(
            temperature=getattr(
                percept,
                "temperature",
                None
            ),
            heart_rate=getattr(
                percept,
                "heart_rate",
                None
            )
        )

        # ----------------------------------------------------
        # 4. Forward chaining
        # ----------------------------------------------------

        inferred = self.forward_chain(
            verbose=False
        )

        # ----------------------------------------------------
        # 5. Get top diagnosis
        # ----------------------------------------------------

        diagnosis, confidence = (
            self.get_top_diagnosis(
                inferred
            )
        )

        return {
            "summary": (
                f"Inferred {len(inferred)} "
                f"facts/conclusions"
            ),

            "diagnosis": diagnosis,

            "confidence": round(
                confidence,
                4
            ),

            "inferred_facts": inferred,

            "inference_history":
                list(self.inference_history)
        }

    # ========================================================
    # EXPLANATION
    # ========================================================

    def get_explanation(
        self,
        diagnosis: str
    ) -> str:
        """
        Explain the rule that can produce a diagnosis.
        """

        diagnosis = self.normalize_fact(
            diagnosis
        )

        for conditions, conclusion, certainty in self.rules:

            if conclusion == diagnosis:

                return (
                    f"'{diagnosis}' derived from: "
                    f"{' + '.join(conditions)} "
                    f"(Rule CF={certainty:.2f})"
                )

        return (
            f"'{diagnosis}' is a base fact "
            f"or no direct rule was found."
        )

    # ========================================================
    # DISPLAY KNOWLEDGE BASE
    # ========================================================

    def print_knowledge_base(self):
        """
        Display all medical rules.
        """

        print("\nMedical Knowledge Base")
        print("=" * 70)

        for number, (
            conditions,
            conclusion,
            certainty
        ) in enumerate(
            self.rules,
            start=1
        ):

            print(
                f"{number:02d}. "
                f"{' ∧ '.join(conditions)} "
                f"→ {conclusion} "
                f"(CF={certainty:.2f})"
            )


# ============================================================
# MODULE 2 INDEPENDENT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODULE 2 — MEDICAL KNOWLEDGE BASE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Create knowledge base
    # --------------------------------------------------------

    kb = MedicalKnowledgeBase()

    # --------------------------------------------------------
    # Display loaded rules
    # --------------------------------------------------------

    kb.print_knowledge_base()

    # --------------------------------------------------------
    # Load symptoms from PDF example
    # --------------------------------------------------------

    print("\nLoading patient symptoms...")

    kb.load_patient_symptoms([
        "Fever",
        "Cough",
        "Loss of Smell",
        "Fatigue"
    ])

    # --------------------------------------------------------
    # Display facts
    # --------------------------------------------------------

    print("\nKnown Facts")
    print("-" * 60)

    for fact in sorted(kb.facts):

        print(
            f"{fact}: "
            f"CF={kb.certainty_factors[fact]:.3f}"
        )

    # --------------------------------------------------------
    # Forward chaining
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FORWARD CHAINING")
    print("=" * 60)

    inferred = kb.forward_chain(
        verbose=True
    )

    print("\nInferred facts:")

    for fact, confidence in inferred.items():

        print(
            f"{fact}: "
            f"CF={confidence:.3f}"
        )

    # --------------------------------------------------------
    # Backward chaining
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BACKWARD CHAINING")
    print("=" * 60)

    proved, confidence = (
        kb.backward_chain(
            "covid19_suspected"
        )
    )

    print(
        f"COVID-19 suspected: "
        f"{proved}, "
        f"Confidence: "
        f"{confidence:.3f}"
    )

    # --------------------------------------------------------
    # Top diagnosis
    # --------------------------------------------------------

    diagnosis, confidence = (
        kb.get_top_diagnosis(
            inferred
        )
    )

    print("\nTop Diagnosis")
    print("-" * 60)

    print(
        f"{diagnosis} "
        f"(CF={confidence:.3f})"
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    print("\nExplanation")
    print("-" * 60)

    print(
        kb.get_explanation(
            diagnosis
        )
    )

    print("\nModule 2 test completed.")