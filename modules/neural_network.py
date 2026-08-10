# ============================================================
# MODULE 5: Deep Neural Network Diagnostic Model
# Covers: Week 10 (Neural Networks & Deep Learning)
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from typing import Dict, List
from tensorflow.keras import layers, models, callbacks


class NeuralDiagnosticModel:
    """
    Deep Neural Network for medical diagnosis.

    Architecture:
        Input (18)
            ↓
        Dense (128) + ReLU + BatchNorm + Dropout(30%)
            ↓
        Dense (64) + ReLU + BatchNorm + Dropout(20%)
            ↓
        Dense (32) + ReLU + BatchNorm
            ↓
        Output (8) + Softmax
    """

    # --------------------------------------------------------
    # 18 symptom features required by the project
    # --------------------------------------------------------

    SYMPTOM_FEATURES = [
        'fever',
        'cough',
        'fatigue',
        'headache',
        'body_aches',
        'loss_of_smell',
        'chest_pain',
        'rash',
        'joint_pain',
        'shortness_of_breath',
        'sweating',
        'frequent_urination',
        'excessive_thirst',
        'blurred_vision',
        'night_sweats',
        'weight_loss',
        'stiff_neck',
        'light_sensitivity'
    ]

    # --------------------------------------------------------
    # 8 disease classes
    # --------------------------------------------------------

    DISEASE_LABELS = [
        'flu',
        'covid19',
        'dengue',
        'cardiac_event',
        'diabetes',
        'common_cold',
        'tuberculosis',
        'meningitis'
    ]

    def __init__(self):
        self.model = None
        self.history = None
        self.is_trained = False

        self._build_model()

    # ========================================================
    # STEP 1: BUILD NEURAL NETWORK
    # ========================================================

    def _build_model(self):
        """Build the required deep neural network architecture."""

        n_inputs = len(self.SYMPTOM_FEATURES)
        n_outputs = len(self.DISEASE_LABELS)

        self.model = models.Sequential([

            # Input Layer
            layers.Input(shape=(n_inputs,)),

            # Hidden Layer 1
            layers.Dense(
                128,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.30),

            # Hidden Layer 2
            layers.Dense(
                64,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.20),

            # Hidden Layer 3
            layers.Dense(
                32,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),

            # Output Layer
            layers.Dense(
                n_outputs,
                activation='softmax'
            )

        ], name='MedicalDNN')

        # ----------------------------------------------------
        # Compile model
        # ----------------------------------------------------

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # ========================================================
    # STEP 2: GENERATE SYNTHETIC DATA
    # ========================================================

    def _generate_data(self, n: int = 3000):
        """
        Generate synthetic patient data.

        Each patient is represented by an 18-dimensional
        binary symptom vector.
        """

        np.random.seed(42)

        profiles = {

            'flu': {
                'fever': 0.90,
                'cough': 0.85,
                'fatigue': 0.88,
                'headache': 0.70,
                'body_aches': 0.80
            },

            'covid19': {
                'fever': 0.88,
                'cough': 0.80,
                'fatigue': 0.90,
                'loss_of_smell': 0.85,
                'headache': 0.65
            },

            'dengue': {
                'fever': 0.98,
                'rash': 0.75,
                'joint_pain': 0.85,
                'headache': 0.90,
                'fatigue': 0.80
            },

            'cardiac_event': {
                'chest_pain': 0.92,
                'shortness_of_breath': 0.88,
                'sweating': 0.75,
                'fatigue': 0.70
            },

            'diabetes': {
                'fatigue': 0.82,
                'frequent_urination': 0.95,
                'excessive_thirst': 0.92,
                'blurred_vision': 0.70
            },

            'common_cold': {
                'cough': 0.90,
                'fever': 0.50,
                'headache': 0.60,
                'fatigue': 0.55
            },

            'tuberculosis': {
                'cough': 0.95,
                'weight_loss': 0.85,
                'night_sweats': 0.80,
                'fatigue': 0.88,
                'fever': 0.70
            },

            'meningitis': {
                'headache': 0.95,
                'stiff_neck': 0.90,
                'fever': 0.92,
                'light_sensitivity': 0.85
            }
        }

        X_list = []
        y_list = []

        n_per_disease = n // len(profiles)

        for label_index, (disease, probabilities) in enumerate(
                profiles.items()):

            for _ in range(n_per_disease):

                row = []

                for feature in self.SYMPTOM_FEATURES:

                    probability = probabilities.get(
                        feature,
                        0.03
                    )

                    value = (
                        1.0
                        if np.random.random() < probability
                        else 0.0
                    )

                    row.append(value)

                X_list.append(row)
                y_list.append(label_index)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        # Shuffle the complete dataset
        indices = np.random.permutation(len(X))

        X = X[indices]
        y = y[indices]

        return X, y

    # ========================================================
    # STEP 3: TRAIN MODEL
    # ========================================================

    def train(self, epochs: int = 30, verbose: int = 1) -> Dict:
        """
        Train the neural network using synthetic patient data.

        Default is 30 epochs for practical testing.
        """

        X, y = self._generate_data(3000)

        # ----------------------------------------------------
        # 80% training / 20% validation
        # ----------------------------------------------------

        split = int(0.80 * len(X))

        X_train = X[:split]
        X_val = X[split:]

        y_train = y[:split]
        y_val = y[split:]

        self._X_val = X_val
        self._y_val = y_val

        # ----------------------------------------------------
        # Training callbacks
        # ----------------------------------------------------

        callback_list = [

            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),

            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        print("=" * 60)
        print("MODULE 5 — DEEP NEURAL NETWORK DIAGNOSTIC MODEL")
        print("=" * 60)

        print("\nNetwork Architecture:")
        print(
            f"Input:          {len(self.SYMPTOM_FEATURES)} neurons"
        )
        print("Hidden Layer 1: 128 neurons + BatchNorm + Dropout(30%)")
        print("Hidden Layer 2: 64 neurons + BatchNorm + Dropout(20%)")
        print("Hidden Layer 3: 32 neurons + BatchNorm")
        print(
            f"Output:         {len(self.DISEASE_LABELS)} neurons + Softmax"
        )

        print("\nTraining Dataset:")
        print(f"Total samples:  {len(X)}")
        print(f"Training:       {len(X_train)}")
        print(f"Validation:     {len(X_val)}")

        print("\nTraining model...")
        print("-" * 60)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=callback_list,
            verbose=verbose
        )

        # ----------------------------------------------------
        # Training completed
        # ----------------------------------------------------

        best_val_accuracy = max(
            self.history.history['val_accuracy']
        )

        self.is_trained = True

        print("\n" + "=" * 60)
        print(
            f"Best Validation Accuracy: "
            f"{best_val_accuracy:.4f}"
        )
        print("=" * 60)

        return {
            'validation_accuracy': round(
                float(best_val_accuracy), 4
            ),
            'epochs_completed': len(
                self.history.history['loss']
            )
        }

    # ========================================================
    # STEP 4: PREDICT
    # ========================================================

    def predict(self, symptoms: List[str]) -> Dict:
        """
        Predict disease from a list of symptoms.

        Symptoms are converted into an 18-dimensional
        binary feature vector.
        """

        if not self.is_trained:
            self.train(verbose=0)

        # Clean symptoms
        cleaned_symptoms = {
            symptom.lower().strip().replace(' ', '_')
            for symptom in symptoms
        }

        # Create binary feature vector
        features = np.array([
            [
                1.0 if feature in cleaned_symptoms else 0.0
                for feature in self.SYMPTOM_FEATURES
            ]
        ], dtype=np.float32)

        # Neural network prediction
        probabilities = self.model.predict(
            features,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        diagnosis = self.DISEASE_LABELS[predicted_index]

        confidence = float(
            probabilities[predicted_index]
        )

        all_probabilities = dict(
            zip(
                self.DISEASE_LABELS,
                probabilities.tolist()
            )
        )

        return {
            'diagnosis': diagnosis,
            'confidence': round(confidence, 4),
            'all_probs': {
                disease: round(probability, 4)
                for disease, probability
                in all_probabilities.items()
            },
            'symptom_vector': features[0].tolist()
        }

    # ========================================================
    # AGENT INTERFACE
    # ========================================================

    def analyze(self, percept) -> Dict:
        """Standard interface used by Module 1 Agent."""

        result = self.predict(percept.symptoms)

        result['summary'] = (
            f"DNN: {result['diagnosis']} "
            f"({result['confidence']:.2%})"
        )

        return result

    # ========================================================
    # STEP 5: PLOT TRAINING PERFORMANCE
    # ========================================================

    def plot_training(self):
        """
        Plot:
        1. Training vs validation accuracy
        2. Training vs validation loss
        """

        if self.history is None:
            print("Train the model first!")
            return

        history = self.history.history

        # ----------------------------------------------------
        # Accuracy graph
        # ----------------------------------------------------

        plt.figure(figsize=(8, 5))

        plt.plot(
            history['accuracy'],
            label='Training Accuracy'
        )

        plt.plot(
            history['val_accuracy'],
            label='Validation Accuracy'
        )

        plt.title("Neural Network Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "nn_accuracy.png",
            dpi=150,
            bbox_inches='tight'
        )

        plt.show()

        # ----------------------------------------------------
        # Loss graph
        # ----------------------------------------------------

        plt.figure(figsize=(8, 5))

        plt.plot(
            history['loss'],
            label='Training Loss'
        )

        plt.plot(
            history['val_loss'],
            label='Validation Loss'
        )

        plt.title("Neural Network Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "nn_loss.png",
            dpi=150,
            bbox_inches='tight'
        )

        plt.show()

        print("\nSaved:")
        print("  nn_accuracy.png")
        print("  nn_loss.png")


# ============================================================
# MODULE 5 TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("MODULE 5 — DEEP NEURAL NETWORK TEST")
    print("=" * 60)

    # Create model
    nn = NeuralDiagnosticModel()

    # Train
    training_result = nn.train(
        epochs=30,
        verbose=1
    )

    # Test patient
    symptoms = [
        "fever",
        "rash",
        "joint_pain",
        "headache"
    ]

    print("\n")
    print("Test Patient Symptoms:")
    print(", ".join(symptoms))

    # Predict
    result = nn.predict(symptoms)

    print("\n")
    print("Diagnosis")
    print("-" * 40)
    print(f"Diagnosis : {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")

    print("\nAll Disease Probabilities")
    print("-" * 40)

    for disease, probability in sorted(
        result['all_probs'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{disease:<20}: {probability:.2%}")

    print("\nSymptom Vector")
    print("-" * 40)
    print(result['symptom_vector'])

    # Generate training graphs
    print("\nGenerating training graphs...")
    nn.plot_training()

    print("\n")
    print("=" * 60)
    print("Module 5 test completed.")
    print("=" * 60)