# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# Introduction to AI — 13-Week Capstone Project
# ============================================================

import warnings

warnings.filterwarnings("ignore")

# ============================================================
# IMPORT AI MODULES
# ============================================================

from modules.agent import (
    HealthcareDiagnosticAgent,
    PatientPercept
)

from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner import TreatmentPlanner


# ============================================================
# ANSI TERMINAL COLORS
# ============================================================

class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def banner():
    print(f"""
{C.BOLD}{C.BLUE}
╔══════════════════════════════════════════════════════════╗
║       🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║
║        Introduction to AI — Capstone Project            ║
║                                                          ║
║  Agent | Logic | Bayes | ML | DNN | Fuzzy | Planning    ║
╚══════════════════════════════════════════════════════════╝
{C.END}
""")


def section(title):
    print(
        f"\n{C.BOLD}{C.YELLOW}"
        + "═" * 60
        + f"{C.END}"
    )

    print(
        f"{C.BOLD}{C.YELLOW}  {title}{C.END}"
    )

    print(
        f"{C.BOLD}{C.YELLOW}"
        + "═" * 60
        + f"{C.END}"
    )


# ============================================================
# BUILD AI SYSTEM
# ============================================================

def build_system():
    """
    Create the Intelligent Healthcare Diagnostic Agent
    and register Modules 2–7.
    """

    section("🔧 BUILDING AI SYSTEM")

    print("\n  Initializing AI modules...")

    agent = HealthcareDiagnosticAgent()

    # --------------------------------------------------------
    # Module 2 — Knowledge Base
    # --------------------------------------------------------

    print("  ✓ Initializing KnowledgeBase")

    knowledge_base = MedicalKnowledgeBase()

    agent.register_module(
        "KnowledgeBase",
        knowledge_base
    )

    # --------------------------------------------------------
    # Module 3 — Bayesian Diagnostics
    # --------------------------------------------------------

    print("  ✓ Initializing BayesianNet")

    bayesian = SimpleBayesianDiagnostics()

    agent.register_module(
        "BayesianNet",
        bayesian
    )

    # --------------------------------------------------------
    # Module 4 — Machine Learning
    # --------------------------------------------------------

    print("  ✓ Initializing MLClassifier")

    ml_classifier = MLDiagnosticClassifier()

    agent.register_module(
        "MLClassifier",
        ml_classifier
    )

    # --------------------------------------------------------
    # Module 5 — Deep Neural Network
    # --------------------------------------------------------

    print("  ✓ Initializing NeuralNetwork")

    neural_network = NeuralDiagnosticModel()

    agent.register_module(
        "NeuralNetwork",
        neural_network
    )

    # --------------------------------------------------------
    # Module 6 — Fuzzy Logic
    # --------------------------------------------------------

    print("  ✓ Initializing FuzzySeverity")

    fuzzy_controller = FuzzySeverityAssessor()

    agent.register_module(
        "FuzzySeverity",
        fuzzy_controller
    )

    # --------------------------------------------------------
    # Module 7 — AI Planning
    # --------------------------------------------------------

    print("  ✓ Initializing TreatmentPlanner")

    planner = TreatmentPlanner()

    agent.register_module(
        "TreatmentPlanner",
        planner
    )

    print(
        f"\n{C.GREEN}  ✓ All AI modules registered successfully."
        f"{C.END}"
    )

    return agent


# ============================================================
# DISPLAY PATIENT
# ============================================================

def display_patient(patient):
    """
    Display patient information.
    """

    print(f"""
Patient ID       : {patient.patient_id}
Age              : {patient.age}
Temperature      : {patient.temperature} °C
Heart Rate       : {patient.heart_rate} BPM
Blood Pressure   : {patient.blood_pressure}
Symptoms         : {", ".join(patient.symptoms)}
""")


# ============================================================
# DISPLAY MODULE RESULTS
# ============================================================

def display_module_results(results):
    """
    Display results returned by each AI module.
    """

    section("🤖 AI MODULE RESULTS")

    for module_name, result in results.items():

        print(f"\n{C.BOLD}{module_name}{C.END}")
        print("-" * 50)

        if not isinstance(result, dict):
            print(result)
            continue

        # Diagnosis
        if "diagnosis" in result:
            print(
                f"Diagnosis   : {result['diagnosis']}"
            )

        # Confidence
        if "confidence" in result:

            try:
                print(
                    f"Confidence  : "
                    f"{float(result['confidence']):.2%}"
                )

            except (ValueError, TypeError):
                print(
                    f"Confidence  : {result['confidence']}"
                )

        # Severity
        if "severity_score" in result:
            print(
                f"Severity    : "
                f"{result['severity_score']}/100"
            )

        if "severity_label" in result:
            print(
                f"Severity    : "
                f"{result['severity_label']}"
            )

        # Treatment plan
        if "steps" in result:
            print(
                f"Plan Steps  : {result['steps']}"
            )

        if "total_duration" in result:
            print(
                f"Duration    : {result['total_duration']}"
            )

        # Summary
        if "summary" in result:
            print(
                f"Summary     : {result['summary']}"
            )

        # Errors
        if "error" in result:
            print(
                f"{C.RED}Error       : "
                f"{result['error']}{C.END}"
            )


# ============================================================
# DISPLAY FINAL AGENT REPORT
# ============================================================

def display_final_report(report):
    """
    Display the final report produced by the intelligent agent.
    """

    section("📋 FINAL AGENT ASSESSMENT")

    print(
        f"Patient ID       : "
        f"{report.get('patient_id', 'N/A')}"
    )

    print(
        f"Diagnosis        : "
        f"{report.get('diagnosis', 'N/A')}"
    )

    print(
        f"Confidence       : "
        f"{report.get('confidence', 0):.2%}"
    )

    print(
        f"Urgency          : "
        f"{report.get('urgency', 'N/A')}"
    )

    print(
        f"Next Action      : "
        f"{report.get('next_action', 'N/A')}"
    )

    print("\nRecommendations:")
    print("-" * 50)

    for recommendation in report.get(
        "recommendations",
        []
    ):
        print(f"  • {recommendation}")

    print(
        f"\nAgent State      : "
        f"{report.get('agent_state', 'N/A')}"
    )


# ============================================================
# TEST PATIENTS
# ============================================================

def create_test_patients():
    """
    Create five synthetic patients representing
    different diagnostic scenarios.

    These are demonstration patients only.
    """

    return [

        # ----------------------------------------------------
        # Patient 1 — COVID-19-like presentation
        # ----------------------------------------------------

        PatientPercept(
            patient_id="P001",
            age=34,
            temperature=38.9,
            heart_rate=98,
            blood_pressure="120/80",
            symptoms=[
                "fever",
                "cough",
                "fatigue",
                "loss_of_smell"
            ]
        ),

        # ----------------------------------------------------
        # Patient 2 — Dengue-like presentation
        # ----------------------------------------------------

        PatientPercept(
            patient_id="P002",
            age=27,
            temperature=39.2,
            heart_rate=105,
            blood_pressure="118/76",
            symptoms=[
                "fever",
                "rash",
                "joint_pain",
                "headache",
                "fatigue"
            ]
        ),

        # ----------------------------------------------------
        # Patient 3 — Diabetes-like presentation
        # ----------------------------------------------------

        PatientPercept(
            patient_id="P003",
            age=45,
            temperature=37.1,
            heart_rate=82,
            blood_pressure="125/82",
            symptoms=[
                "fatigue",
                "frequent_urination",
                "excessive_thirst",
                "blurred_vision",
                "weight_loss"
            ]
        ),

        # ----------------------------------------------------
        # Patient 4 — Cardiac-event-like presentation
        # ----------------------------------------------------

        PatientPercept(
            patient_id="P004",
            age=59,
            temperature=37.4,
            heart_rate=118,
            blood_pressure="150/95",
            symptoms=[
                "chest_pain",
                "shortness_of_breath",
                "sweating",
                "fatigue"
            ]
        ),

        # ----------------------------------------------------
        # Patient 5 — Tuberculosis-like presentation
        # ----------------------------------------------------

        PatientPercept(
            patient_id="P005",
            age=41,
            temperature=38.4,
            heart_rate=102,
            blood_pressure="122/80",
            symptoms=[
                "cough",
                "fatigue",
                "weight_loss",
                "night_sweats",
                "fever"
            ]
        )
    ]


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_patient(agent, patient):
    """
    Run one patient through the complete
    Perceive → Think → Act cycle.
    """

    section(f"👤 PATIENT {patient.patient_id}")

    display_patient(patient)

    print(
        f"{C.CYAN}  Perceiving patient..."
        f"{C.END}"
    )

    print(
        f"{C.CYAN}  Running AI reasoning modules..."
        f"{C.END}"
    )

    print(
        f"{C.CYAN}  Generating final assessment..."
        f"{C.END}"
    )

    # ========================================================
    # IMPORTANT:
    #
    # run() internally performs:
    #
    # perceive()
    #     ↓
    # think()
    #     ↓
    # act(diagnosis_results)
    #
    # This fixes the previous missing diagnosis_results error.
    # ========================================================

    report = agent.run(patient)

    # Display individual module results
    display_module_results(
        report.get("module_results", {})
    )

    # Display final agent report
    display_final_report(report)

    return report


# ============================================================
# FIVE-PATIENT SUMMARY
# ============================================================

def display_summary(results):
    """
    Display summary of the five-patient test.
    """

    section("📊 FIVE-PATIENT TEST SUMMARY")

    if not results:
        print(
            f"{C.RED}"
            "No patients were successfully processed."
            f"{C.END}"
        )
        return

    print(
        f"\n{'Patient':<10}"
        f"{'Diagnosis':<20}"
        f"{'Confidence':<15}"
        f"{'Urgency':<12}"
    )

    print("-" * 57)

    for result in results:

        patient_id = result.get(
            "patient_id",
            "N/A"
        )

        diagnosis = result.get(
            "diagnosis",
            "N/A"
        )

        confidence = result.get(
            "confidence",
            0
        )

        urgency = result.get(
            "urgency",
            "N/A"
        )

        print(
            f"{patient_id:<10}"
            f"{diagnosis:<20}"
            f"{confidence:<15.2%}"
            f"{urgency:<12}"
        )


# ============================================================
# AGENT PERFORMANCE
# ============================================================

def display_agent_performance(agent):
    """
    Display basic Agent performance information.
    """

    section("📈 AGENT PERFORMANCE")

    performance = agent.get_performance()

    print(
        f"Total Patients Processed : "
        f"{performance['total_patients']}"
    )

    print(
        f"Diagnoses Made           : "
        f"{performance['diagnoses_made']}"
    )

    print(
        f"Registered Modules       : "
        f"{performance['registered_modules']}"
    )

    print(
        f"Performance Score        : "
        f"{performance['performance_score']}"
    )

    print(
        f"Current Agent State       : "
        f"{performance['current_state']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Banner
    # --------------------------------------------------------

    banner()

    # --------------------------------------------------------
    # Build system
    # --------------------------------------------------------

    agent = build_system()

    # --------------------------------------------------------
    # Create five synthetic patients
    # --------------------------------------------------------

    patients = create_test_patients()

    section("🧪 FULL SYSTEM TEST — 5 PATIENTS")

    successful_results = []

    # --------------------------------------------------------
    # Process each patient
    # --------------------------------------------------------

    for patient in patients:

        try:

            result = process_patient(
                agent,
                patient
            )

            successful_results.append(
                result
            )

        except Exception as error:

            print(
                f"\n{C.RED}"
                f"❌ Error while processing "
                f"{patient.patient_id}:"
                f"{C.END}"
            )

            print(
                f"  {type(error).__name__}: "
                f"{error}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    display_summary(
        successful_results
    )

    # --------------------------------------------------------
    # Agent performance
    # --------------------------------------------------------

    display_agent_performance(
        agent
    )

    # --------------------------------------------------------
    # Completion message
    # --------------------------------------------------------

    section("✅ CAPSTONE INTEGRATION TEST COMPLETE")

    print("""
  Modules integrated:

    ✓ Module 1 — Intelligent Agent
    ✓ Module 2 — Knowledge Base
    ✓ Module 3 — Bayesian Diagnostics
    ✓ Module 4 — Machine Learning
    ✓ Module 5 — Neural Network
    ✓ Module 6 — Fuzzy Severity
    ✓ Module 7 — Treatment Planning

  Five-patient end-to-end test completed.

  Next stage:
    → Evaluation metrics
    → Confusion matrices
    → Model comparison
    → Final report
    → README update
""")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()