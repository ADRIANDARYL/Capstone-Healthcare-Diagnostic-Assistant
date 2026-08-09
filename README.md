# Intelligent Healthcare Diagnostic Assistant

## Introduction to Artificial Intelligence — Capstone Project

An academic AI healthcare prototype that combines multiple Artificial Intelligence techniques to analyse patient information, estimate severity, and generate a treatment-planning sequence.

> **Important:** This project is for academic and demonstration purposes only. It is not a substitute for professional medical diagnosis, triage, treatment, or clinical decision-making.

---

## Project Overview

The Intelligent Healthcare Diagnostic Assistant uses a modular AI architecture in which different AI techniques perform specialized tasks while an intelligent agent coordinates the overall workflow.

The system follows the:

**Perceive → Think → Act**

cycle.

### Main Workflow

1. Patient information is represented as a `PatientPercept`.
2. The intelligent agent perceives and stores the patient information.
3. Registered AI modules analyse the patient.
4. Diagnostic results and confidence values are collected.
5. The agent determines an overall diagnosis and urgency.
6. The fuzzy controller estimates severity.
7. The treatment planner generates a sequence of actions.
8. A structured final assessment is produced.

---

# AI Modules

| Module | Component | AI Concept | Main Purpose |
|---|---|---|---|
| 1 | `modules/agent.py` | Intelligent Agents / PEAS | Coordinates the complete AI workflow |
| 2 | `modules/knowledge_base.py` | Knowledge Representation | Stores and provides medical knowledge |
| 3 | `modules/bayesian_net.py` | Bayesian Reasoning | Performs probabilistic diagnosis |
| 4 | `modules/ml_classifier.py` | Machine Learning | Classifies patient cases |
| 5 | `modules/neural_network.py` | Deep Learning | Performs neural-network diagnosis |
| 6 | `modules/fuzzy_controller.py` | Fuzzy Logic | Estimates patient severity |
| 7 | `modules/planner.py` | AI Planning / STRIPS / BFS | Generates step-by-step treatment plans |

---

# Module 1 — Intelligent Agent

The intelligent agent coordinates the other AI components.

### Agent Architecture

```text
Patient
   |
   v
PERCEIVE
   |
   v
THINK
   |
   +---- Knowledge Base
   +---- Bayesian Diagnostics
   +---- ML Classifier
   +---- Neural Network
   +---- Fuzzy Severity
   +---- Treatment Planner
   |
   v
ACT
   |
   v
Final Assessment
```

The agent maintains:

- Patient history
- Current patient percept
- Diagnostic history
- Action log
- Registered AI modules
- Current agent state

The agent uses a **model-based and goal-based** approach.

---

# Module 2 — Knowledge Base

The knowledge-base module provides structured medical information used by the diagnostic system.

It supports the system's knowledge representation component and provides information that can be used when interpreting patient symptoms and diagnoses.

---

# Module 3 — Bayesian Diagnostics

The Bayesian diagnostic module applies probabilistic reasoning to patient symptoms.

It estimates the likelihood of possible diagnoses based on the available evidence.

This demonstrates:

- Prior probabilities
- Evidence
- Conditional probability
- Posterior probability
- Probabilistic diagnostic reasoning

---

# Module 4 — Machine Learning Classifier

The ML diagnostic module trains a machine-learning classifier using the project's diagnostic dataset.

The module supports:

- Training
- Validation
- Prediction
- Diagnostic analysis
- Model evaluation
- Confusion-matrix generation

---

# Module 5 — Deep Neural Network

The neural diagnostic module uses a deep neural network for disease classification.

### Network Architecture

```text
Input:          18 neurons
Hidden Layer 1: 128 neurons + BatchNorm + Dropout
Hidden Layer 2: 64 neurons + BatchNorm + Dropout
Hidden Layer 3: 32 neurons + BatchNorm
Output:         8 neurons + Softmax
```

The model is trained and evaluated independently before being used by the complete diagnostic agent.

---

# Module 6 — Fuzzy Severity Assessment

The fuzzy controller determines the severity of a patient's condition.

It uses three inputs:

- Temperature
- Heart rate
- Number of symptoms

### Fuzzy Logic Process

#### 1. Fuzzification

Crisp patient values are converted into membership degrees.

#### 2. Rule Evaluation

Fuzzy rules are evaluated using operations such as `min()` for AND and `max()` for OR.

#### 3. Defuzzification

The fuzzy result is converted into a severity score from 0 to 100.

### Severity Categories

| Score | Label |
|---:|---|
| 0–19 | LOW |
| 20–39 | MILD |
| 40–59 | MODERATE |
| 60–79 | HIGH |
| 80–100 | CRITICAL |

---

# Module 7 — AI Treatment Planner

The treatment planner demonstrates AI planning using **STRIPS-style actions**.

Each action contains:

- Preconditions
- Delete list
- Add list
- Cost
- Duration

Example:

```text
Action: OrderPCRTest

Preconditions:
    COVID_SUSPECTED
    PATIENT_PRESENT

Delete:
    COVID_SUSPECTED

Add:
    PCR_PENDING
```

The planner uses **Breadth-First Search (BFS)** to search through possible states until the goal state is reached.

Typical goal predicates include:

```text
TREATMENT_STARTED
VITALS_MONITORED
FOLLOWUP_SCHEDULED
```

---

# System Integration

The complete system is started through:

```text
app.py
```

The application creates and registers all seven AI modules with the healthcare diagnostic agent.

### System Flow

```text
Patient Data
     |
     v
PatientPercept
     |
     v
HealthcareDiagnosticAgent
     |
     +----------------------+
     |                      |
     v                      v
Diagnostic Modules      Severity Module
     |                      |
     +----------+-----------+
                |
                v
        Diagnosis + Severity
                |
                v
        Treatment Planner
                |
                v
        Final Assessment
```

---

# Running the Project

## 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-directory>
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

## 3. Activate the Virtual Environment

On Windows:

```bash
venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Complete Application

```bash
python app.py
```

The application runs the integrated AI workflow and tests the system using five patient cases.

---

# Independent Module Testing

Individual modules can also be tested independently.

For example:

```bash
python modules\agent.py
```

Evaluation modules can be checked using:

```bash
python evaluation\metrics.py
python evaluation\visualizations.py
```

The evaluation pipeline can be executed using:

```bash
python run_evaluation.py
```

---

# Model Evaluation

The evaluation pipeline consists of:

```text
evaluation/
├── metrics.py
├── visualizations.py
└── results/
```

The evaluation focuses on:

- Module 4 — ML Classifier
- Module 5 — Neural Network

The evaluator calculates:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC when probability predictions are available

---

# Evaluation Results

The current evaluation used **8 test patients**.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Module 4 — ML Classifier | 87.50% | 81.25% | 87.50% | 83.33% | N/A |
| Module 5 — Neural Network | 100.00% | 100.00% | 100.00% | 100.00% | N/A |

### Interpretation

The ML classifier achieved an accuracy of **87.5%** on the current evaluation data.

The neural network achieved **100%** on the current evaluation set.

The neural-network result should be interpreted only as performance on the project's evaluation data. It does **not** represent perfect real-world clinical accuracy.

ROC-AUC is currently reported as **N/A** because probability-score arrays were not supplied to the evaluation process.

---

# Evaluation Outputs

The evaluation process generates:

```text
evaluation/
└── results/
    ├── module_4_-_ml_classifier_confusion_matrix.png
    ├── module_5_-_neural_network_confusion_matrix.png
    ├── model_comparison.png
    └── evaluation_results.csv
```

### Confusion Matrices

The confusion matrices compare:

- Actual diagnoses
- Predicted diagnoses

for each evaluated classifier.

### Model Comparison

`model_comparison.png` provides a visual comparison of:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

# Full-System Test

The integrated application was successfully tested using **five patient cases**.

The patient cases included different combinations of:

- Age
- Temperature
- Heart rate
- Blood pressure
- Symptoms

The full system successfully integrated:

```text
✓ Module 1 — Intelligent Agent
✓ Module 2 — Knowledge Base
✓ Module 3 — Bayesian Diagnostics
✓ Module 4 — Machine Learning
✓ Module 5 — Neural Network
✓ Module 6 — Fuzzy Severity
✓ Module 7 — Treatment Planning
```

---

# Project Structure

```text
pdf-capstone/
│
├── app.py
├── run_evaluation.py
├── README.md
├── requirements.txt
│
├── modules/
│   ├── __init__.py
│   ├── agent.py
│   ├── knowledge_base.py
│   ├── bayesian_net.py
│   ├── ml_classifier.py
│   ├── neural_network.py
│   ├── fuzzy_controller.py
│   └── planner.py
│
├── evaluation/
│   ├── metrics.py
│   ├── visualizations.py
│   └── results/
│       ├── evaluation_results.csv
│       ├── model_comparison.png
│       ├── module_4_-_ml_classifier_confusion_matrix.png
│       └── module_5_-_neural_network_confusion_matrix.png
│
├── reports/
│   └── AI_Healthcare_Diagnostic_Assistant_Final_Report.pdf
│
└── venv/
```

> `venv/` is a local Python virtual environment and should normally be excluded from Git using `.gitignore`.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- ReportLab
- Python Virtual Environment

---

# AI Concepts Demonstrated

| AI Concept | Application |
|---|---|
| Intelligent Agents | Healthcare diagnostic agent |
| PEAS | Agent performance, environment, actuators and sensors |
| Knowledge Representation | Medical knowledge base |
| Bayesian Reasoning | Probabilistic diagnosis |
| Machine Learning | Diagnostic classification |
| Neural Networks | Deep diagnostic model |
| Fuzzy Logic | Patient severity assessment |
| Search Algorithms | BFS-based treatment planning |
| AI Planning | STRIPS-style treatment actions |
| Model Evaluation | Classification metrics and confusion matrices |

---

# Limitations

This system has several important limitations:

1. It is an academic prototype and has not been clinically validated.
2. The learned models depend on the available training and evaluation data.
3. Synthetic or limited datasets may not represent real-world patient populations.
4. A high evaluation score does not guarantee real-world clinical performance.
5. The treatment planner demonstrates AI planning rather than an actual clinical treatment protocol.
6. The system should not be used to make real medical decisions.
7. ROC-AUC is currently unavailable because probability predictions were not supplied to the evaluation process.

---

# Future Improvements

Possible future improvements include:

- Use larger clinically validated datasets.
- Add probability outputs for ROC-AUC evaluation.
- Perform cross-validation and more extensive testing.
- Add additional diagnostic conditions.
- Improve the treatment-planning knowledge base.
- Add a graphical/web user interface.
- Add authentication and patient-history management.
- Add explainable-AI features showing why a diagnosis was selected.
- Add more comprehensive safety and clinical validation procedures.

---

# Final Deliverables

Current project status:

- [x] Module 1 — Intelligent Agent
- [x] Module 2 — Knowledge Base
- [x] Module 3 — Bayesian Diagnostics
- [x] Module 4 — ML Classifier
- [x] Module 5 — Neural Network
- [x] Module 6 — Fuzzy Severity Assessor
- [x] Module 7 — AI Treatment Planner
- [x] Modules integrated into `app.py`
- [x] Five-patient full-system test
- [x] Evaluation metrics
- [x] Confusion matrices
- [x] Model comparison chart
- [x] Evaluation CSV
- [x] Final report PDF
- [x] Final 10-minute live demonstration

---

# Academic Note

This project demonstrates the integration of multiple Artificial Intelligence techniques into a single healthcare-oriented application. Its purpose is to demonstrate AI concepts, system integration, model evaluation, and reasoning techniques in an academic setting.