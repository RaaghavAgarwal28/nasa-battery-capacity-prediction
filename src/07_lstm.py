"""
=========================================================
NASA Battery Capacity Prediction
Phase 5 : LSTM (Sequential Model)
=========================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add current folder to path to allow import of utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import prepare_data

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs")
MODEL_OUT_FOLDER = os.path.join(OUTPUT_FOLDER, "lstm")

MODELLING_DATASET_FILE = os.path.join(OUTPUT_FOLDER, "modelling_dataset.csv")

os.makedirs(MODEL_OUT_FOLDER, exist_ok=True)

print("=" * 60)
print("NASA Battery Capacity Prediction")
print("Phase 5 : LSTM Regressor")
print("=" * 60)

# =========================================================
# PREPARE DATA
# =========================================================

print("\nPreparing and scaling sequence dataset...")
data = prepare_data(MODELLING_DATASET_FILE, seq_length=5, test_size=0.2, random_state=42)

X_train = data['X_train_seq']
X_test = data['X_test_seq']
y_train = data['y_train']
y_test = data['y_test']

print(f"Train set shape (Sequences): {X_train.shape}")
print(f"Test set shape  (Sequences): {X_test.shape}")

N_samples, seq_length, num_features = X_train.shape

# =========================================================
# BUILD LSTM MODEL
# =========================================================

print("\nBuilding LSTM Model...")
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(seq_length, num_features), return_sequences=False),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1)  # Linear activation for regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining LSTM Model...")
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=60,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
print("Model training complete.")

# Predict
y_pred = model.predict(X_test).flatten()

# =========================================================
# EVALUATE MODEL
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nLSTM Test Metrics:")
print(f"MAE     : {mae:.6f}")
print(f"MSE     : {mse:.6f}")
print(f"RMSE    : {rmse:.6f}")
print(f"R² Score: {r2:.6f}")

# Save metrics CSV
metrics_df = pd.DataFrame({
    'Model': ['LSTM'],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'R2': [r2]
})
metrics_path = os.path.join(MODEL_OUT_FOLDER, "metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\nMetrics saved to: {metrics_path}")

# =========================================================
# GENERATE PLOTS
# =========================================================

# 1. Loss Curve
fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
ax.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=1.5)
if 'val_loss' in history.history:
    ax.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=1.5)
ax.set_title("LSTM Model Loss (MSE) vs Epochs", fontsize=14, fontweight='bold')
ax.set_xlabel("Epochs", fontsize=12)
ax.set_ylabel("Loss (MSE)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
loss_path = os.path.join(MODEL_OUT_FOLDER, "loss_curve.png")
fig.savefig(loss_path, dpi=300)
plt.close(fig)
print(f"Saved loss curve: {loss_path}")

# 2. Actual vs Predicted (Line plot showing degradation profiles)
sort_idx = np.argsort(y_test)
fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
ax.plot(np.arange(len(y_test)), y_test[sort_idx], color='black', label='Actual', linewidth=1.5)
ax.plot(np.arange(len(y_test)), y_pred[sort_idx], color='red', label='Predicted', linewidth=1.0, alpha=0.8)
ax.set_title("Actual vs Predicted Capacity (LSTM)", fontsize=14, fontweight='bold')
ax.set_xlabel("Sorted Test Samples", fontsize=12)
ax.set_ylabel("Capacity (Ah)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
fig.savefig(os.path.join(MODEL_OUT_FOLDER, "actual_vs_predicted.png"), dpi=300)
plt.close(fig)

# 3. Residual Plot (Residual vs Predicted)
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
ax.scatter(y_pred, residuals, color='purple', alpha=0.6, edgecolors='black', s=20)
ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_title("Residual Plot (LSTM)", fontsize=14, fontweight='bold')
ax.set_xlabel("Predicted Capacity (Ah)", fontsize=12)
ax.set_ylabel("Residuals (Ah)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
fig.savefig(os.path.join(MODEL_OUT_FOLDER, "residual_plot.png"), dpi=300)
plt.close(fig)

# 4. Prediction Error Plot (Actual vs Predicted Scatter)
fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
ax.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolors='black', s=20)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit')
ax.set_title("Prediction Error Plot (LSTM)", fontsize=14, fontweight='bold')
ax.set_xlabel("Actual Capacity (Ah)", fontsize=12)
ax.set_ylabel("Predicted Capacity (Ah)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
ax.set_aspect('equal', adjustable='box')
fig.savefig(os.path.join(MODEL_OUT_FOLDER, "prediction_error.png"), dpi=300)
plt.close(fig)

print("Plots generated and saved successfully.")

print("\n" + "=" * 60)
print("LSTM Model Completed Successfully")
print("=" * 60)
