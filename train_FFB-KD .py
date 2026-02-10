import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# --- 1. DATA PREPROCESSING ---
# Ensure your dataset is in the same directory
df = pd.read_csv('dataset_cleaned_smote.csv')
X = df.drop('Label_Num', axis=1).values
y = df['Label_Num'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]

# --- 2. MODEL ARCHITECTURES ---
# For Feature-Based Distillation, we name intermediate layers as "hints"
def build_teacher(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    # This is our 'Hint' layer (Intermediate features)
    hint_layer = layers.Dense(64, activation='relu', name="teacher_feature")(x)
    outputs = layers.Dense(1, activation='sigmoid')(hint_layer)
    return keras.Model(inputs, outputs, name="Teacher_5Layer")

def build_student(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    # This is our 'Guided' layer (Student's version of features)
    student_feature = layers.Dense(32, activation='relu', name="student_feature")(inputs)
    outputs = layers.Dense(1, activation='sigmoid')(student_feature)
    return keras.Model(inputs, outputs, name="Student_3Layer")

# --- 3. THE FEATURE-BASED DISTILLER ---
class FeatureDistiller(keras.Model):
    def __init__(self, student, teacher):
        super(FeatureDistiller, self).__init__()
        self.teacher = teacher
        self.student = student

        # Create models that output BOTH intermediate features and final predictions
        self.teacher_feat_model = keras.Model(
            inputs=teacher.inputs,
            outputs=[teacher.get_layer("teacher_feature").output, teacher.output]
        )
        self.student_feat_model = keras.Model(
            inputs=student.inputs,
            outputs=[student.get_layer("student_feature").output, student.output]
        )

        # LINEAR PROJECTOR: Maps Student's 32 neurons to Teacher's 64 neurons
        self.projector = layers.Dense(64)

    def compile(self, optimizer, metrics, alpha=0.1, gamma=0.7):
        super(FeatureDistiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.alpha = alpha   # Weight for Classification Loss
        self.gamma = gamma   # Weight for Feature Alignment Loss
        self.ce_loss_fn = keras.losses.BinaryCrossentropy()
        self.mse_loss_fn = keras.losses.MeanSquaredError()

    def train_step(self, data):
        x, y = data

        # 1. Get Teacher's internal features (Hints)
        teacher_features, _ = self.teacher_feat_model(x, training=False)

        with tf.GradientTape() as tape:
            # 2. Get Student's internal features and predictions
            student_features, student_predictions = self.student_feat_model(x, training=True)

            # 3. Align Student features to Teacher feature dimensions
            projected_student_features = self.projector(student_features)

            # 4. Calculate Losses
            # Task Loss (Classification)
            loss_ce = self.ce_loss_fn(y, student_predictions)

            # Feature Alignment Loss (MSE between Teacher and Projected Student features)
            loss_feat = self.mse_loss_fn(teacher_features, projected_student_features)

            # Total Loss = Task + Feature Alignment
            total_loss = (self.alpha * loss_ce) + (self.gamma * loss_feat)

        # 5. Optimize only the Student and Projector variables
        trainable_vars = self.student.trainable_variables + self.projector.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.compiled_metrics.update_state(y, student_predictions)
        return {m.name: m.result() for m in self.metrics}

# --- 4. EXECUTION FLOW ---
print("Training Teacher Model...")
teacher_model = build_teacher(input_dim)
teacher_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
teacher_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

print("Training Feature-Based Student (FBKD)...")
fbkd_student = build_student(input_dim)
distiller = FeatureDistiller(student=fbkd_student, teacher=teacher_model)
distiller.compile(optimizer='adam', metrics=['accuracy'], alpha=0.3, gamma=0.7)
distiller.fit(X_train, y_train, epochs=15, batch_size=64, verbose=0)

# --- 5. TFLITE INT8 QUANTIZATION ---
def representative_data_gen():
    for i in range(100):
        yield [X_train[i].astype(np.float32).reshape(1, -1)]

def quantize_model(model, name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    path = f"{name}.tflite"
    with open(path, "wb") as f: f.write(tflite_model)
    return path

print("Quantizing FBKD Model to INT8...")
fbkd_q_path = quantize_model(fbkd_student, "fbkd_student_int8")

# --- 6. RESULTS & SUMMARY ---
size_kb = os.path.getsize(fbkd_q_path) / 1024
print(f"\n--- FBKD RESULTS ---")
print(f"Quantized Model Path: {fbkd_q_path}")
print(f"Final Model Size: {size_kb:.2f} KB") # Aiming for ~5.26 KB