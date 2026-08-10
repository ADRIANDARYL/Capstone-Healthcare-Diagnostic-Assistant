# ============================================================
# MODULE 7: AI Planning — Treatment Plan Generator
# Covers: Week 12 (AI Planning Techniques)
# ============================================================

from collections import deque
from typing import Dict, List, Set, Optional


class TreatmentPlanner:
    """
    STRIPS-based treatment planner.

    Generates step-by-step treatment plans using:
    - Preconditions
    - Delete lists
    - Add lists
    - Breadth-First Search (BFS)

    NOTE:
    This is an educational AI planning simulation and
    not a real-world medical treatment recommendation system.
    """

    def __init__(self):
        self.action_library = self._build_action_library()

    # ========================================================
    # ACTION LIBRARY
    # ========================================================

    def _build_action_library(self) -> List[Dict]:
        """Define available planning actions."""

        return [

            # ------------------------------------------------
            # EMERGENCY ACTIONS
            # ------------------------------------------------

            {
                'name': 'CallEmergencyServices',
                'precond': {
                    'EMERGENCY_CASE',
                    'PATIENT_PRESENT'
                },
                'delete': {
                    'EMERGENCY_CASE'
                },
                'add': {
                    'EMERGENCY_SERVICES_CALLED'
                },
                'cost': 0,
                'duration': '5 minutes'
            },

            {
                'name': 'TransferToICU',
                'precond': {
                    'EMERGENCY_SERVICES_CALLED',
                    'ICU_AVAILABLE'
                },
                'delete': {
                    'EMERGENCY_SERVICES_CALLED'
                },
                'add': {
                    'PATIENT_IN_ICU',
                    'MONITORING_ACTIVE'
                },
                'cost': 0,
                'duration': '15 minutes'
            },

            # ------------------------------------------------
            # DIAGNOSTIC ACTIONS
            # ------------------------------------------------

            {
                'name': 'OrderBloodPanel',
                'precond': {
                    'PATIENT_PRESENT',
                    'DIAGNOSIS_NEEDED'
                },
                'delete': {
                    'DIAGNOSIS_NEEDED'
                },
                'add': {
                    'BLOOD_RESULTS_PENDING'
                },
                'cost': 1,
                'duration': '30 minutes'
            },

            {
                'name': 'ReceiveBloodResults',
                'precond': {
                    'BLOOD_RESULTS_PENDING'
                },
                'delete': {
                    'BLOOD_RESULTS_PENDING'
                },
                'add': {
                    'BLOOD_RESULTS_AVAILABLE',
                    'DIAGNOSIS_REFINED',
                    'DIAGNOSIS_CONFIRMED'
                },
                'cost': 0,
                'duration': '2 hours'
            },

            {
                'name': 'OrderPCRTest',
                'precond': {
                    'COVID_SUSPECTED',
                    'PATIENT_PRESENT'
                },
                'delete': {
                    'COVID_SUSPECTED'
                },
                'add': {
                    'PCR_PENDING'
                },
                'cost': 1,
                'duration': '24 hours'
            },

            {
                'name': 'ReceivePCRResult',
                'precond': {
                    'PCR_PENDING'
                },
                'delete': {
                    'PCR_PENDING'
                },
                'add': {
                    'PCR_RESULT_AVAILABLE',
                    'DIAGNOSIS_CONFIRMED',
                    'VIRAL_INFECTION'
                },
                'cost': 0,
                'duration': '24 hours'
            },

            # ------------------------------------------------
            # GENERAL DIAGNOSIS CONFIRMATION
            # ------------------------------------------------

            {
                'name': 'ConfirmDiagnosis',
                'precond': {
                    'DIAGNOSIS_NEEDED'
                },
                'delete': {
                    'DIAGNOSIS_NEEDED'
                },
                'add': {
                    'DIAGNOSIS_CONFIRMED'
                },
                'cost': 1,
                'duration': '30 minutes'
            },

            # ------------------------------------------------
            # TREATMENT ACTIONS
            # ------------------------------------------------

            {
                'name': 'PrescribeAntiviral',
                'precond': {
                    'DIAGNOSIS_CONFIRMED',
                    'VIRAL_INFECTION'
                },
                'delete': {
                    'VIRAL_INFECTION'
                },
                'add': {
                    'ANTIVIRAL_PRESCRIBED',
                    'TREATMENT_STARTED'
                },
                'cost': 1,
                'duration': '10 minutes'
            },

            {
                'name': 'PrescribeAntibiotics',
                'precond': {
                    'DIAGNOSIS_CONFIRMED',
                    'BACTERIAL_INFECTION'
                },
                'delete': {
                    'BACTERIAL_INFECTION'
                },
                'add': {
                    'ANTIBIOTICS_PRESCRIBED',
                    'TREATMENT_STARTED'
                },
                'cost': 1,
                'duration': '10 minutes'
            },

            {
                'name': 'AdministerFluids',
                'precond': {
                    'PATIENT_IN_ICU',
                    'DEHYDRATION_RISK'
                },
                'delete': {
                    'DEHYDRATION_RISK'
                },
                'add': {
                    'FLUIDS_ADMINISTERED'
                },
                'cost': 1,
                'duration': '1 hour'
            },

            {
                'name': 'StartSupportiveTreatment',
                'precond': {
                    'DIAGNOSIS_CONFIRMED'
                },
                'delete': set(),
                'add': {
                    'TREATMENT_STARTED'
                },
                'cost': 1,
                'duration': '15 minutes'
            },

            # ------------------------------------------------
            # ISOLATION
            # ------------------------------------------------

            {
                'name': 'IsolatePatient',
                'precond': {
                    'CONTAGIOUS_DISEASE',
                    'PATIENT_PRESENT'
                },
                'delete': {
                    'CONTAGIOUS_DISEASE'
                },
                'add': {
                    'PATIENT_ISOLATED'
                },
                'cost': 0,
                'duration': '14 days'
            },

            # ------------------------------------------------
            # MONITORING
            # ------------------------------------------------

            {
                'name': 'MonitorVitals',
                'precond': {
                    'TREATMENT_STARTED',
                    'PATIENT_PRESENT'
                },
                'delete': set(),
                'add': {
                    'VITALS_MONITORED'
                },
                'cost': 0,
                'duration': 'Continuous'
            },

            # ------------------------------------------------
            # FOLLOW-UP
            # ------------------------------------------------

            {
                'name': 'ScheduleFollowUp',
                'precond': {
                    'TREATMENT_STARTED',
                    'VITALS_MONITORED'
                },
                'delete': set(),
                'add': {
                    'FOLLOWUP_SCHEDULED',
                    'PLAN_COMPLETE'
                },
                'cost': 0,
                'duration': '5 minutes'
            },

            # ------------------------------------------------
            # DISCHARGE
            # ------------------------------------------------

            {
                'name': 'DischargePatient',
                'precond': {
                    'PLAN_COMPLETE',
                    'SYMPTOMS_RESOLVED'
                },
                'delete': {
                    'PLAN_COMPLETE'
                },
                'add': {
                    'PATIENT_DISCHARGED'
                },
                'cost': 0,
                'duration': '30 minutes'
            },
        ]

    # ========================================================
    # APPLY STRIPS ACTION
    # ========================================================

    def _apply_action(
        self,
        state: frozenset,
        action: Dict
    ) -> Optional[frozenset]:
        """
        Apply an action if all preconditions are satisfied.

        STRIPS:
        New State = (Current State - Delete List) ∪ Add List
        """

        if not action['precond'].issubset(state):
            return None

        new_state = (
            state - action['delete']
        ) | action['add']

        return frozenset(new_state)

    # ========================================================
    # BFS PLANNER
    # ========================================================

    def generate_plan(
        self,
        initial_state: Set[str],
        goal_state: Set[str]
    ) -> Optional[List[Dict]]:
        """
        Generate a plan using Breadth-First Search.

        BFS explores the state space level by level and
        returns the first plan that reaches the goal.
        """

        initial = frozenset(initial_state)
        goal = frozenset(goal_state)

        queue = deque([
            (initial, [])
        ])

        visited = {initial}

        while queue:

            state, plan = queue.popleft()

            # Check whether goal has been reached
            if goal.issubset(state):
                return plan

            # Try every available action
            for action in self.action_library:

                new_state = self._apply_action(
                    state,
                    action
                )

                if (
                    new_state is not None
                    and new_state not in visited
                ):

                    visited.add(new_state)

                    queue.append(
                        (
                            new_state,
                            plan + [action]
                        )
                    )

        return None

    # ========================================================
    # CREATE TREATMENT PLAN
    # ========================================================

    def create_treatment_plan(
        self,
        diagnosis: str,
        urgency: str
    ) -> Dict:
        """
        Create a treatment plan based on diagnosis
        and urgency.
        """

        diagnosis_key = (
            diagnosis.lower()
            .replace(' ', '_')
        )

        urgency = urgency.upper()

        # ----------------------------------------------------
        # Diagnosis → Initial State
        # ----------------------------------------------------

        diagnosis_states = {

            'flu': {
                'VIRAL_INFECTION',
                'DIAGNOSIS_NEEDED'
            },

            'covid19': {
                'COVID_SUSPECTED',
                'CONTAGIOUS_DISEASE',
                'DIAGNOSIS_NEEDED'
            },

            'cardiac_event': {
                'EMERGENCY_CASE',
                'ICU_AVAILABLE'
            },

            'dengue': {
                'VIRAL_INFECTION',
                'DIAGNOSIS_NEEDED',
                'DEHYDRATION_RISK'
            },

            'meningitis': {
                'EMERGENCY_CASE',
                'BACTERIAL_INFECTION',
                'ICU_AVAILABLE'
            },

            'tuberculosis': {
                'BACTERIAL_INFECTION',
                'CONTAGIOUS_DISEASE',
                'DIAGNOSIS_NEEDED'
            },

            'diabetes': {
                'DIAGNOSIS_NEEDED'
            },

            'common_cold': {
                'VIRAL_INFECTION',
                'DIAGNOSIS_NEEDED'
            }
        }

        # ----------------------------------------------------
        # Base patient state
        # ----------------------------------------------------

        initial_state = {
            'PATIENT_PRESENT'
        }

        initial_state |= diagnosis_states.get(
            diagnosis_key,
            {'DIAGNOSIS_NEEDED'}
        )

        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        goal_state = {
            'TREATMENT_STARTED',
            'VITALS_MONITORED',
            'FOLLOWUP_SCHEDULED'
        }

        # Critical patients must reach ICU
        if urgency == 'CRITICAL':
            goal_state.add('PATIENT_IN_ICU')

        # ----------------------------------------------------
        # Generate plan
        # ----------------------------------------------------

        plan = self.generate_plan(
            initial_state,
            goal_state
        )

        # No solution
        if plan is None:
            return {
                'error': 'No valid treatment plan found',
                'diagnosis': diagnosis,
                'urgency': urgency,
                'initial_state': sorted(initial_state),
                'goal_state': sorted(goal_state),
                'steps': 0,
                'plan': []
            }

        # ----------------------------------------------------
        # Return formatted plan
        # ----------------------------------------------------

        return {
            'diagnosis': diagnosis,
            'urgency': urgency,
            'initial_state': sorted(initial_state),
            'goal_state': sorted(goal_state),
            'steps': len(plan),
            'total_duration': self._estimate_duration(plan),

            'plan': [
                {
                    'step': i + 1,
                    'action': action['name'],
                    'duration': action['duration'],
                    'cost': action['cost']
                }

                for i, action in enumerate(plan)
            ]
        }

    # ========================================================
    # ESTIMATE PLAN DURATION
    # ========================================================

    def _estimate_duration(
        self,
        plan: List[Dict]
    ) -> str:
        """
        Return a simple description of the plan duration.
        """

        durations = [
            action['duration']
            for action in plan
        ]

        return (
            f"{len(plan)} actions | "
            f"See individual durations"
        )

    # ========================================================
    # AGENT INTERFACE
    # ========================================================

    def analyze(self, percept) -> Dict:
        """
        Module interface for the intelligent agent.

        Uses diagnosis/urgency from the percept when available.
        Falls back to demonstration values for standalone testing.
        """

        diagnosis = getattr(
            percept,
            'diagnosis',
            'flu'
        )

        urgency = getattr(
            percept,
            'urgency',
            'MEDIUM'
        )

        result = self.create_treatment_plan(
            diagnosis,
            urgency
        )

        # Handle planning failure
        if 'error' in result:
            result['summary'] = (
                f"No treatment plan found for "
                f"{diagnosis}"
            )
            result['confidence'] = 0.0
            return result

        result['summary'] = (
            f"Plan: {result['steps']} "
            f"steps generated for {diagnosis}"
        )

        result['confidence'] = 0.9

        return result


# ============================================================
# STANDALONE MODULE 7 TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODULE 7 — AI TREATMENT PLANNER TEST")
    print("=" * 60)

    planner = TreatmentPlanner()

    # Test COVID-19
    print("\nCOVID-19 — HIGH")
    print("-" * 40)

    result = planner.create_treatment_plan(
        'covid19',
        'HIGH'
    )

    if 'error' in result:

        print("ERROR:", result['error'])

    else:

        print("Diagnosis :", result['diagnosis'])
        print("Urgency   :", result['urgency'])
        print("Plan Steps:", result['steps'])
        print()

        for step in result['plan']:

            print(
                f"Step {step['step']:2d}: "
                f"{step['action']:<30} "
                f"[{step['duration']}]"
            )

    # --------------------------------------------------------
    # Test cardiac emergency
    # --------------------------------------------------------

    print("\nCARDIAC EVENT — CRITICAL")
    print("-" * 40)

    result = planner.create_treatment_plan(
        'cardiac_event',
        'CRITICAL'
    )

    if 'error' in result:

        print("ERROR:", result['error'])

    else:

        print("Diagnosis :", result['diagnosis'])
        print("Urgency   :", result['urgency'])
        print("Plan Steps:", result['steps'])
        print()

        for step in result['plan']:

            print(
                f"Step {step['step']:2d}: "
                f"{step['action']:<30} "
                f"[{step['duration']}]"
            )

    print("\nModule 7 test completed.")