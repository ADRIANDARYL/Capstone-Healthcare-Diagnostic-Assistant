import streamlit as st
import sys
import os
from datetime import datetime

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# IMPORTS
# ============================================================

from modules.agent import (
    HealthcareDiagnosticAgent,
    PatientPercept
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Healthcare Diagnostic Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .header {
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e3a8a
        );
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
    }

    .header h1 {
        margin-bottom: 5px;
        font-size: 36px;
    }

    .header p {
        margin: 0;
        font-size: 16px;
        color: #dbeafe;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
    }

    .diagnosis-card {
        background: #f8fafc;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #2563eb;
        margin-top: 20px;
    }

    .metric-card {
        background: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    .warning-box {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }

    .critical-box {
        background: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }

    .success-box {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="header">
        <h1>🏥 AI Healthcare Diagnostic Assistant</h1>
        <p>
            Intelligent diagnostic support using multiple AI techniques
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏥 Healthcare AI")

    st.markdown("---")

    st.subheader("Navigation")

    page = st.radio(
        "Select a section",
        [
            "🩺 Diagnosis",
            "📊 Agent Performance",
            "ℹ️ About System"
        ]
    )

    st.markdown("---")

    st.caption(
        "AI Capstone Healthcare Diagnostic Assistant"
    )

    st.caption(
        "Educational demonstration system."
    )

# ============================================================
# CREATE AGENT
# ============================================================

@st.cache_resource
def create_agent():

    agent = HealthcareDiagnosticAgent()

    return agent


agent = create_agent()

# ============================================================
# DIAGNOSIS PAGE
# ============================================================

if page == "🩺 Diagnosis":

    st.subheader("Patient Information")

    st.write(
        "Enter the patient's information below and run the "
        "AI diagnostic assistant."
    )

    # --------------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        patient_id = st.text_input(
            "Patient ID",
            value="P001"
        )

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=30,
            step=1
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=30.0,
            max_value=45.0,
            value=37.0,
            step=0.1
        )

    with col2:

        heart_rate = st.number_input(
            "Heart Rate (BPM)",
            min_value=30,
            max_value=220,
            value=75,
            step=1
        )

        blood_pressure = st.text_input(
            "Blood Pressure",
            value="120/80"
        )

    # --------------------------------------------------------
    # SYMPTOMS
    # --------------------------------------------------------

    st.subheader("Symptoms")

    symptoms_options = [
        "fever",
        "cough",
        "fatigue",
        "headache",
        "sore throat",
        "shortness of breath",
        "chest pain",
        "nausea",
        "vomiting",
        "diarrhea",
        "abdominal pain",
        "loss of smell",
        "loss of taste",
        "muscle pain",
        "joint pain",
        "dizziness",
        "runny nose",
        "congestion"
    ]

    symptoms = st.multiselect(
        "Select symptoms",
        symptoms_options
    )

    additional_symptom = st.text_input(
        "Other symptom (optional)"
    )

    if additional_symptom.strip():

        symptoms.append(
            additional_symptom.strip()
        )

    # --------------------------------------------------------
    # DISPLAY INPUT SUMMARY
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("Patient Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric(
            "Age",
            age
        )

    with summary_col2:
        st.metric(
            "Temperature",
            f"{temperature:.1f} °C"
        )

    with summary_col3:
        st.metric(
            "Heart Rate",
            f"{heart_rate} BPM"
        )

    with summary_col4:
        st.metric(
            "Symptoms",
            len(symptoms)
        )

    # --------------------------------------------------------
    # DIAGNOSE BUTTON
    # --------------------------------------------------------

    st.markdown("---")

    diagnose = st.button(
        "🔍 Run AI Diagnosis",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # RUN DIAGNOSIS
    # ========================================================

    if diagnose:

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not patient_id.strip():

            st.error(
                "Please enter a Patient ID."
            )

            st.stop()

        if not symptoms:

            st.error(
                "Please select at least one symptom."
            )

            st.stop()

        # ----------------------------------------------------
        # IMPORTANT FIX
        #
        # HealthcareDiagnosticAgent.run()
        # REQUIRES PatientPercept.
        #
        # We therefore create a PatientPercept object here
        # instead of passing a dictionary.
        # ----------------------------------------------------

        patient = PatientPercept(

            patient_id=patient_id.strip(),

            symptoms=list(symptoms),

            age=int(age),

            temperature=float(temperature),

            heart_rate=int(heart_rate),

            blood_pressure=blood_pressure.strip(),

            timestamp=datetime.now().isoformat()

        )

        # ----------------------------------------------------
        # RUN AGENT
        # ----------------------------------------------------

        with st.spinner(
            "AI agent is analyzing the patient..."
        ):

            try:

                result = agent.run(patient)

            except Exception as error:

                st.error(
                    "An error occurred during diagnosis."
                )

                st.exception(error)

                st.stop()

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.success(
            "Diagnosis completed successfully."
        )

        st.markdown("---")

        st.subheader("Diagnostic Result")

        # ----------------------------------------------------
        # MAIN RESULT
        # ----------------------------------------------------

        diagnosis = result.get(
            "diagnosis",
            "Insufficient data"
        )

        confidence = result.get(
            "confidence",
            0
        )

        urgency = result.get(
            "urgency",
            "LOW"
        )

        next_action = result.get(
            "next_action",
            "MONITOR_AND_FOLLOWUP"
        )

        # ----------------------------------------------------
        # DIAGNOSIS CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="diagnosis-card">

                <h2>🩺 {diagnosis}</h2>

                <p>
                    <strong>Patient:</strong>
                    {patient_id}
                </p>

                <p>
                    <strong>Confidence:</strong>
                    {confidence * 100:.1f}%
                </p>

                <p>
                    <strong>Urgency:</strong>
                    {urgency}
                </p>

                <p>
                    <strong>Next Action:</strong>
                    {next_action}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # URGENCY DISPLAY
        # ----------------------------------------------------

        if urgency == "CRITICAL":

            st.markdown(
                """
                <div class="critical-box">

                <strong>🚨 CRITICAL URGENCY</strong>

                <br><br>

                Immediate medical assessment is recommended.
                Alert the responsible healthcare professional
                according to appropriate clinical procedures.

                </div>
                """,
                unsafe_allow_html=True
            )

        elif urgency == "HIGH":

            st.markdown(
                """
                <div class="warning-box">

                <strong>⚠️ HIGH URGENCY</strong>

                <br><br>

                Prompt clinical assessment is recommended.
                Review the patient's symptoms and vital signs.

                </div>
                """,
                unsafe_allow_html=True
            )

        elif urgency == "MEDIUM":

            st.info(
                "🟡 MEDIUM URGENCY — "
                "Clinical follow-up is recommended."
            )

        else:

            st.markdown(
                """
                <div class="success-box">

                <strong>🟢 LOW URGENCY</strong>

                <br><br>

                Continue monitoring symptoms and follow up
                if the condition persists or worsens.

                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        st.subheader("📋 System Recommendations")

        recommendations = result.get(
            "recommendations",
            []
        )

        if recommendations:

            for recommendation in recommendations:

                st.write(
                    f"• {recommendation}"
                )

        else:

            st.write(
                "No recommendations were generated."
            )

        # ----------------------------------------------------
        # MODULE RESULTS
        # ----------------------------------------------------

        st.subheader(
            "🤖 AI Module Results"
        )

        module_results = result.get(
            "module_results",
            {}
        )

        if module_results:

            for module_name, module_result in module_results.items():

                with st.expander(
                    f"🔹 {module_name}"
                ):

                    if isinstance(
                        module_result,
                        dict
                    ):

                        module_diagnosis = module_result.get(
                            "diagnosis",
                            "N/A"
                        )

                        module_confidence = module_result.get(
                            "confidence",
                            None
                        )

                        summary = module_result.get(
                            "summary",
                            ""
                        )

                        st.write(
                            f"**Diagnosis:** {module_diagnosis}"
                        )

                        if isinstance(
                            module_confidence,
                            (int, float)
                        ):

                            st.write(
                                f"**Confidence:** "
                                f"{module_confidence * 100:.1f}%"
                            )

                        if summary:

                            st.write(
                                f"**Summary:** {summary}"
                            )

                        # Show errors if a module failed

                        if "error" in module_result:

                            st.error(
                                module_result["error"]
                            )

                    else:

                        st.write(
                            module_result
                        )

        # ----------------------------------------------------
        # PATIENT DATA
        # ----------------------------------------------------

        with st.expander(
            "👤 Patient Data Used"
        ):

            st.json(
                {
                    "patient_id": patient.patient_id,
                    "symptoms": patient.symptoms,
                    "age": patient.age,
                    "temperature": patient.temperature,
                    "heart_rate": patient.heart_rate,
                    "blood_pressure": patient.blood_pressure,
                    "timestamp": patient.timestamp
                }
            )

        # ----------------------------------------------------
        # AGENT STATE
        # ----------------------------------------------------

        with st.expander(
            "⚙️ Agent Information"
        ):

            st.write(
                f"**Agent State:** "
                f"{result.get('agent_state', 'N/A')}"
            )

            st.write(
                f"**Registered Modules:** "
                f"{len(agent.get_registered_modules())}"
            )

            st.write(
                "**Modules:**"
            )

            for module in agent.get_registered_modules():

                st.write(
                    f"• {module}"
                )

# ============================================================
# PERFORMANCE PAGE
# ============================================================

elif page == "📊 Agent Performance":

    st.subheader(
        "📊 Agent Performance"
    )

    performance = agent.get_performance()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Patients Processed",
            performance.get(
                "total_patients",
                0
            )
        )

    with col2:

        st.metric(
            "Diagnoses Made",
            performance.get(
                "diagnoses_made",
                0
            )
        )

    with col3:

        st.metric(
            "Performance Score",
            performance.get(
                "performance_score",
                0
            )
        )

    with col4:

        st.metric(
            "Registered Modules",
            performance.get(
                "registered_modules",
                0
            )
        )

    st.markdown("---")

    st.subheader(
        "Current Agent State"
    )

    st.info(
        performance.get(
            "current_state",
            "idle"
        )
    )

    # --------------------------------------------------------
    # REGISTERED MODULES
    # --------------------------------------------------------

    st.subheader(
        "Registered AI Modules"
    )

    registered_modules = agent.get_registered_modules()

    if registered_modules:

        for module in registered_modules:

            st.success(
                f"✓ {module}"
            )

    else:

        st.warning(
            "No AI modules are currently registered."
        )

    # --------------------------------------------------------
    # ACTION LOG
    # --------------------------------------------------------

    st.subheader(
        "Agent Action Log"
    )

    if agent.memory.action_log:

        for entry in reversed(
            agent.memory.action_log
        ):

            st.write(
                f"• {entry}"
            )

    else:

        st.info(
            "No actions have been recorded yet."
        )

# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About System":

    st.subheader(
        "ℹ️ About the System"
    )

    st.markdown(
        """
        ### AI Healthcare Diagnostic Assistant

        This system is an educational AI capstone project
        demonstrating the integration of multiple Artificial
        Intelligence techniques into a healthcare diagnostic
        assistant.

        ### Intelligent Agent

        The central healthcare agent follows the:

        **Perceive → Think → Act**

        architecture.

        The agent coordinates multiple AI modules and
        combines their outputs into a structured diagnostic
        report.

        ### AI Techniques

        The system contains modules covering:

        - Intelligent Agents
        - Knowledge-Based Systems
        - Bayesian Networks
        - Machine Learning
        - Neural Networks
        - Fuzzy Logic
        - AI Planning

        ### Important Notice

        This application is an educational demonstration.

        It is **not a medical device** and should not be used
        as a substitute for diagnosis, treatment, or advice
        from a qualified healthcare professional.
        """
    )

    st.markdown("---")

    st.info(
        "AI Capstone Healthcare Diagnostic Assistant"
    )