# ============================================================
# MODULE 1: Intelligent Agent — Healthcare Diagnostic Agent
# Covers: Intelligent Agents + PEAS Framework
# Architecture: Perceive → Think → Act
# ============================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import Counter
import datetime


# ============================================================
# 1. AGENT STATE
# ============================================================

class AgentState(Enum):
    """
    Represents the current state of the healthcare agent.
    """

    IDLE = "idle"
    COLLECTING = "collecting_symptoms"
    DIAGNOSING = "diagnosing"
    RECOMMENDING = "recommending"
    PLANNING = "planning_treatment"
    DONE = "done"


# ============================================================
# 2. PATIENT PERCEPT
# ============================================================

@dataclass
class PatientPercept:
    """
    Represents information perceived by the agent
    from the patient/environment.
    """

    patient_id: str
    symptoms: List[str]
    age: int
    temperature: float
    heart_rate: int
    blood_pressure: str

    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )


# ============================================================
# 3. AGENT MEMORY
# ============================================================

@dataclass
class AgentMemory:
    """
    Internal memory of the agent.

    This supports the model-based nature of the agent by
    allowing it to retain information about patients,
    previous diagnostic results and actions.
    """

    patient_history: List[Dict[str, Any]] = field(
        default_factory=list
    )

    current_patient: Optional[PatientPercept] = None

    diagnosis_history: List[Dict[str, Any]] = field(
        default_factory=list
    )

    action_log: List[str] = field(
        default_factory=list
    )


# ============================================================
# 4. HEALTHCARE DIAGNOSTIC AGENT
# ============================================================

class HealthcareDiagnosticAgent:
    """
    Intelligent Healthcare Diagnostic Agent.

    PEAS FRAMEWORK
    ----------------------------------------------------------
    Performance:
        Diagnostic accuracy, patient safety,
        recommendation quality and response time.

    Environment:
        Hospital/clinic environment, patient data
        and electronic medical records.

    Actuators:
        Diagnosis report, recommendations,
        treatment-planning request and alerts.

    Sensors:
        Symptoms, vital signs, laboratory results
        and patient history.

    AGENT TYPE
    ----------------------------------------------------------
    Model-Based + Goal-Based Agent

    CORE CYCLE
    ----------------------------------------------------------
    Perceive → Think → Act

    The Agent coordinates the individual AI modules.
    It does not replace the individual diagnostic modules.
    """

    def __init__(self):

        # Current state of the agent
        self.state = AgentState.IDLE

        # Internal memory
        self.memory = AgentMemory()

        # Simple performance tracking
        self.performance_score = 0

        # Registered AI modules
        self._modules: Dict[str, Any] = {}

    # ========================================================
    # MODULE REGISTRATION
    # ========================================================

    def register_module(self, name: str, module: Any) -> None:
        """
        Register an AI sub-module with the agent.

        Examples:
            Knowledge Base
            Bayesian Network
            ML Classifier
            Neural Network
            Fuzzy Controller
            Planner
        """

        if not name:
            raise ValueError("Module name cannot be empty.")

        if module is None:
            raise ValueError(
                f"Module '{name}' cannot be None."
            )

        self._modules[name] = module

        self._log(
            f"Module registered: [{name}]"
        )

    def get_registered_modules(self) -> List[str]:
        """
        Return the names of all registered modules.
        """

        return list(self._modules.keys())

    # ========================================================
    # STEP 1 — PERCEIVE
    # ========================================================

    def perceive(self, percept: PatientPercept):
        """
        Step 1 of the Perceive → Think → Act cycle.

        The agent:
        1. Receives patient information.
        2. Stores the current patient.
        3. Adds the patient to its history.
        4. Records the perception.
        5. Changes state to COLLECTING.
        """

        if not isinstance(percept, PatientPercept):
            raise TypeError(
                "percept must be a PatientPercept object."
            )

        self.memory.current_patient = percept

        # Store patient information in memory
        self.memory.patient_history.append({
            "id": percept.patient_id,
            "symptoms": list(percept.symptoms),
            "age": percept.age,
            "temperature": percept.temperature,
            "heart_rate": percept.heart_rate,
            "blood_pressure": percept.blood_pressure,
            "time": percept.timestamp
        })

        # Update agent state
        self.state = AgentState.COLLECTING

        self._log(
            f"Perceived patient {percept.patient_id} "
            f"with {len(percept.symptoms)} symptoms"
        )

        return self

    # ========================================================
    # STEP 2 — THINK
    # ========================================================

    def think(self) -> Dict[str, Any]:
        """
        Step 2 of the Perceive → Think → Act cycle.

        The agent sends the current patient percept to every
        registered module that provides an analyze() method.

        Each module returns its own diagnostic result.
        """

        if self.memory.current_patient is None:
            raise RuntimeError(
                "No patient has been perceived. "
                "Call perceive() before think()."
            )

        self.state = AgentState.DIAGNOSING

        self._log(
            "Agent thinking: running diagnostic modules..."
        )

        results: Dict[str, Any] = {}

        # Run every registered module
        for module_name, module in self._modules.items():

            # Modules are expected to provide analyze()
            if not hasattr(module, "analyze"):
                self._log(
                    f"[{module_name}] skipped - "
                    f"no analyze() method"
                )
                continue

            try:

                result = module.analyze(
                    self.memory.current_patient
                )

                results[module_name] = result

                # Safely obtain a summary if the result is a dictionary
                if isinstance(result, dict):
                    summary = result.get(
                        "summary",
                        "analysis completed"
                    )
                else:
                    summary = "analysis completed"

                self._log(
                    f"[{module_name}] → {summary}"
                )

            except Exception as error:

                # Keep the Agent running even if one module fails.
                results[module_name] = {
                    "diagnosis": "Module error",
                    "confidence": 0.0,
                    "summary": (
                        f"{module_name} failed during analysis"
                    ),
                    "error": str(error)
                }

                self._log(
                    f"[{module_name}] → analysis failed"
                )

        # Store complete diagnostic results in memory
        self.memory.diagnosis_history.append(results)

        # Move to recommendation stage
        self.state = AgentState.RECOMMENDING

        return results

    # ========================================================
    # STEP 3 — ACT
    # ========================================================

    def act(self, diagnosis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3 of the Perceive → Think → Act cycle.

        The agent:
        1. Aggregates module confidence scores.
        2. Determines urgency.
        3. Determines the most supported diagnosis.
        4. Generates recommendations.
        5. Determines the next action.
        6. Produces a structured report.
        """

        if self.memory.current_patient is None:
            raise RuntimeError(
                "No patient has been perceived. "
                "Call perceive() before act()."
            )

        if not isinstance(diagnosis_results, dict):
            raise TypeError(
                "diagnosis_results must be a dictionary."
            )

        self.state = AgentState.PLANNING

        patient = self.memory.current_patient

        # ----------------------------------------------------
        # Aggregate confidence
        # ----------------------------------------------------

        confidences = []

        for result in diagnosis_results.values():

            if isinstance(result, dict):

                confidence = result.get(
                    "confidence"
                )

                if isinstance(confidence, (int, float)):

                    # Keep confidence within 0 and 1
                    confidence = max(
                        0.0,
                        min(1.0, float(confidence))
                    )

                    confidences.append(confidence)

        if confidences:

            avg_confidence = (
                sum(confidences) / len(confidences)
            )

        else:

            avg_confidence = 0.0

        # ----------------------------------------------------
        # Determine urgency
        # ----------------------------------------------------

        urgency = self._assess_urgency(
            patient,
            avg_confidence
        )

        # ----------------------------------------------------
        # Aggregate diagnosis
        # ----------------------------------------------------

        diagnosis = self._aggregate_diagnosis(
            diagnosis_results
        )

        # ----------------------------------------------------
        # Generate recommendations
        # ----------------------------------------------------

        recommendations = self._generate_recommendations(
            urgency,
            diagnosis_results
        )

        # ----------------------------------------------------
        # Determine next action
        # ----------------------------------------------------

        next_action = self._decide_next_action(
            urgency
        )

        # ----------------------------------------------------
        # Final structured report
        # ----------------------------------------------------

        action_report = {

            "patient_id": patient.patient_id,

            "timestamp": patient.timestamp,

            "symptoms": list(patient.symptoms),

            "diagnosis": diagnosis,

            "confidence": round(
                avg_confidence,
                3
            ),

            "urgency": urgency,

            "recommendations": recommendations,

            "next_action": next_action,

            "module_results": diagnosis_results,

            "agent_state": self.state.value
        }

        # Basic performance tracking
        if avg_confidence >= 0.7:
            self.performance_score += 10
        else:
            self.performance_score += 5

        # Agent has completed its cycle
        self.state = AgentState.DONE

        action_report["agent_state"] = self.state.value

        self._log(
            f"Action generated: {urgency} urgency"
        )

        return action_report

    # ========================================================
    # COMPLETE AGENT CYCLE
    # ========================================================

    def run(
        self,
        percept: PatientPercept
    ) -> Dict[str, Any]:
        """
        Execute the complete:

            Perceive → Think → Act

        cycle.
        """

        self.perceive(percept)

        diagnosis_results = self.think()

        return self.act(
            diagnosis_results
        )

    # ========================================================
    # URGENCY ASSESSMENT
    # ========================================================

    def _assess_urgency(
        self,
        patient: PatientPercept,
        confidence: float
    ) -> str:
        """
        Determine urgency using patient vital information
        and aggregated diagnostic confidence.

        NOTE:
        This is a project/demo decision rule and is not
        intended to replace professional medical triage.
        """

        if (
            patient.temperature > 39.5
            or patient.heart_rate > 120
        ):
            return "CRITICAL"

        elif (
            patient.temperature > 38.5
            or confidence > 0.8
        ):
            return "HIGH"

        elif patient.temperature > 37.5:

            return "MEDIUM"

        return "LOW"

    # ========================================================
    # DIAGNOSIS AGGREGATION
    # ========================================================

    def _aggregate_diagnosis(
        self,
        results: Dict[str, Any]
    ) -> str:
        """
        Combine diagnoses from the registered modules.

        The diagnosis supported by the greatest number of
        modules is selected.
        """

        diagnoses = []

        for result in results.values():

            if isinstance(result, dict):

                diagnosis = result.get(
                    "diagnosis"
                )

                if diagnosis:
                    diagnoses.append(
                        str(diagnosis)
                    )

        if not diagnoses:
            return "Insufficient data"

        # Majority voting
        return Counter(
            diagnoses
        ).most_common(1)[0][0]

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def _generate_recommendations(
        self,
        urgency: str,
        results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate high-level system recommendations based
        on the calculated urgency.

        These are system-level recommendations for the
        capstone demonstration and are not prescriptions.
        """

        recommendations = {

            "CRITICAL": [
                "Immediate medical assessment recommended.",
                "Alert the responsible healthcare professional.",
                "Consider emergency referral according to clinical protocol.",
                "Continue monitoring vital signs."
            ],

            "HIGH": [
                "Prompt clinical assessment recommended.",
                "Review the patient's vital signs and symptoms.",
                "Consider appropriate diagnostic investigations.",
                "Continue close monitoring."
            ],

            "MEDIUM": [
                "Clinical follow-up is recommended.",
                "Continue monitoring symptoms and vital signs.",
                "Review the diagnostic results with a healthcare professional.",
                "Seek further assessment if symptoms worsen."
            ],

            "LOW": [
                "Continue monitoring the patient's symptoms.",
                "Maintain appropriate general care and hydration.",
                "Follow up if symptoms persist or worsen.",
                "Review the assessment with a healthcare professional when appropriate."
            ]
        }

        return recommendations.get(
            urgency,
            recommendations["LOW"]
        )

    # ========================================================
    # NEXT ACTION
    # ========================================================

    def _decide_next_action(
        self,
        urgency: str
    ) -> str:
        """
        Convert urgency into a high-level system action.
        """

        actions = {

            "CRITICAL":
                "EMERGENCY_REFERRAL",

            "HIGH":
                "URGENT_CLINICAL_ASSESSMENT",

            "MEDIUM":
                "SCHEDULE_FOLLOWUP",

            "LOW":
                "MONITOR_AND_FOLLOWUP"
        }

        return actions.get(
            urgency,
            "MONITOR_AND_FOLLOWUP"
        )

    # ========================================================
    # ACTION LOG
    # ========================================================

    def _log(
        self,
        message: str
    ) -> None:
        """
        Store an event in the Agent's internal action log.
        """

        entry = (
            f"[{self.state.value}] {message}"
        )

        self.memory.action_log.append(
            entry
        )

    def print_log(self) -> None:
        """
        Display the Agent's action history.
        """

        print("\nAgent Action Log:")
        print("-" * 50)

        for entry in self.memory.action_log:
            print(f"  {entry}")

    # ========================================================
    # PERFORMANCE INFORMATION
    # ========================================================

    def get_performance(self) -> Dict[str, Any]:
        """
        Return basic information about Agent activity.
        """

        return {

            "total_patients":
                len(
                    self.memory.patient_history
                ),

            "performance_score":
                self.performance_score,

            "diagnoses_made":
                len(
                    self.memory.diagnosis_history
                ),

            "registered_modules":
                len(
                    self._modules
                ),

            "current_state":
                self.state.value
        }


# ============================================================
# MODULE 1 — INDEPENDENT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODULE 1 — INTELLIGENT AGENT TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Temporary test module
    # --------------------------------------------------------

    class TestDiagnosticModule:

        def analyze(
            self,
            patient: PatientPercept
        ) -> Dict[str, Any]:

            return {

                "diagnosis": "Test Diagnosis",

                "confidence": 0.90,

                "summary":
                    "Test diagnostic module completed successfully"
            }

    # --------------------------------------------------------
    # Create patient percept
    # --------------------------------------------------------

    patient = PatientPercept(

        patient_id="P001",

        symptoms=[
            "fever",
            "cough",
            "fatigue",
            "loss of smell"
        ],

        age=34,

        temperature=38.9,

        heart_rate=98,

        blood_pressure="120/80"
    )

    # --------------------------------------------------------
    # Create Agent
    # --------------------------------------------------------

    agent = HealthcareDiagnosticAgent()

    # --------------------------------------------------------
    # Register temporary test module
    # --------------------------------------------------------

    agent.register_module(
        "TestDiagnosticModule",
        TestDiagnosticModule()
    )

    # --------------------------------------------------------
    # Run complete Perceive → Think → Act cycle
    # --------------------------------------------------------

    report = agent.run(patient)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nFinal Report")
    print("-" * 60)

    for key, value in report.items():

        print(f"{key}: {value}")

    # --------------------------------------------------------
    # Display action log
    # --------------------------------------------------------

    agent.print_log()

    # --------------------------------------------------------
    # Display performance
    # --------------------------------------------------------

    print("\nAgent Performance")
    print("-" * 60)

    print(
        agent.get_performance()
    )

    print("\nModule 1 test completed.")