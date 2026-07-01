"""
=========================================================
NASA Battery Capacity Prediction
Phase 9 : Final Report Graphs
=========================================================
Generates comprehensive comparison plots for Linear Regression,
Random Forest, and LSTM models side-by-side for the final report.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import prepare_data

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs")
REPORT_FOLDER = os.path.join(OUTPUT_FOLDER, "report_graphs")
os.makedirs(REPORT_FOLDER, exist_ok=True)

MODELLING_DATASET_FILE = os.path.join(OUTPUT_FOLDER, "modelling_dataset.csv")

print("=" * 60)
print("NASA Battery Capacity Prediction")
print("Phase 9 : Final Report Graphs")
print("=" * 60)

# =========================================================
# PREPARE DATA
# =========================================================
print("\nLoading and preparing data...")
data = prepare_data(MODELLING_DATASET_FILE, seq_length=5, test_size=0.2, random_state=42)

X_train_flat = data['X_train_flat']
X_test_flat  = data['X_test_flat']
X_train_seq  = data['X_train_seq']
X_test_seq   = data['X_test_seq']
y_train      = data['y_train']
y_test       = data['y_test']
feature_names = data['feature_names']

# =========================================================
# TRAIN MODELS
# =========================================================
print("Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train)
y_pred_lr = lr_model.predict(X_test_flat)

print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_flat, y_train)
y_pred_rf = rf_model.predict(X_test_flat)

print("Training LSTM...")
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = lstm_model.fit(
    X_train_seq, y_train,
    epochs=60, batch_size=16,
    validation_split=0.15,
    callbacks=[es],
    verbose=1
)
y_pred_lstm = lstm_model.predict(X_test_seq).flatten()

# =========================================================
# COMPUTE METRICS
# =========================================================
def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

mae_lr,  mse_lr,  rmse_lr,  r2_lr  = metrics(y_test, y_pred_lr)
mae_rf,  mse_rf,  rmse_rf,  r2_rf  = metrics(y_test, y_pred_rf)
mae_lst, mse_lst, rmse_lst, r2_lst = metrics(y_test, y_pred_lstm)

models      = ['Linear\nRegression', 'Random\nForest', 'LSTM']
mae_vals    = [mae_lr,  mae_rf,  mae_lst]
rmse_vals   = [rmse_lr, rmse_rf, rmse_lst]
r2_vals     = [r2_lr,   r2_rf,   r2_lst]
COLORS      = ['#E74C3C', '#2ECC71', '#3498DB']

sort_idx = np.argsort(y_test)

# =========================================================
# HELPER — save and close
# =========================================================
def savefig(fig, name):
    path = os.path.join(REPORT_FOLDER, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# =========================================================
# FIGURE 1 — Actual vs Predicted (all 3 side by side)
# =========================================================
print("\nGenerating Figure 1: Actual vs Predicted...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Actual vs Predicted Battery Capacity", fontsize=16, fontweight='bold', y=1.02)

for ax, (y_pred, label, color) in zip(axes, [
    (y_pred_lr,   'Linear Regression', '#E74C3C'),
    (y_pred_rf,   'Random Forest',     '#2ECC71'),
    (y_pred_lstm, 'LSTM',              '#3498DB'),
]):
    ax.plot(np.arange(len(y_test)), y_test[sort_idx],  color='black', lw=1.8, label='Actual', zorder=3)
    ax.plot(np.arange(len(y_test)), y_pred[sort_idx],  color=color,   lw=1.2, label='Predicted', alpha=0.85, zorder=2)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_xlabel("Sorted Test Samples", fontsize=11)
    ax.set_ylabel("Capacity (Ah)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

fig.tight_layout()
savefig(fig, "fig1_actual_vs_predicted.png")

# =========================================================
# FIGURE 2 — Prediction Error Scatter (all 3 side by side)
# =========================================================
print("Generating Figure 2: Prediction Error Scatter...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Prediction Error — Actual vs Predicted Scatter", fontsize=16, fontweight='bold', y=1.02)

for ax, (y_pred, label, color) in zip(axes, [
    (y_pred_lr,   'Linear Regression', '#E74C3C'),
    (y_pred_rf,   'Random Forest',     '#2ECC71'),
    (y_pred_lstm, 'LSTM',              '#3498DB'),
]):
    ax.scatter(y_test, y_pred, color=color, alpha=0.55, edgecolors='black', s=18)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Fit')
    mae_v, _, rmse_v, r2_v = metrics(y_test, y_pred)
    ax.set_title(f"{label}\nMAE={mae_v:.4f}  RMSE={rmse_v:.4f}  R²={r2_v:.4f}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Actual Capacity (Ah)", fontsize=11)
    ax.set_ylabel("Predicted Capacity (Ah)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_aspect('equal', adjustable='box')

fig.tight_layout()
savefig(fig, "fig2_prediction_error_scatter.png")

# =========================================================
# FIGURE 3 — Residual Plots (all 3 side by side)
# =========================================================
print("Generating Figure 3: Residual Plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Residual Analysis", fontsize=16, fontweight='bold', y=1.02)

for ax, (y_pred, label, color) in zip(axes, [
    (y_pred_lr,   'Linear Regression', '#E74C3C'),
    (y_pred_rf,   'Random Forest',     '#2ECC71'),
    (y_pred_lstm, 'LSTM',              '#3498DB'),
]):
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, color=color, alpha=0.5, edgecolors='black', s=18)
    ax.axhline(0, color='red', linestyle='--', lw=1.8)
    ax.set_title(f"{label}\nResiduals vs Predicted", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Capacity (Ah)", fontsize=11)
    ax.set_ylabel("Residuals (Ah)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)

fig.tight_layout()
savefig(fig, "fig3_residuals.png")

# =========================================================
# FIGURE 4 — Metric Comparison Bar Charts
# =========================================================
print("Generating Figure 4: Metric Comparison Bars...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold', y=1.02)

bar_kw = dict(edgecolor='black', width=0.5, zorder=3)

for ax, (vals, ylabel, title) in zip(axes, [
    (mae_vals,  'MAE (Ah)',   'Mean Absolute Error ↓'),
    (rmse_vals, 'RMSE (Ah)', 'Root Mean Squared Error ↓'),
    (r2_vals,   'R² Score',   'R² Score ↑'),
]):
    bars = ax.bar(models, vals, color=COLORS, **bar_kw)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(True, linestyle='--', alpha=0.35, axis='y', zorder=0)
    ax.set_axisbelow(True)

fig.tight_layout()
savefig(fig, "fig4_metric_comparison.png")

# =========================================================
# FIGURE 5 — LSTM Training Loss Curve
# =========================================================
print("Generating Figure 5: LSTM Training Loss Curve...")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(history.history['loss'],     color='#3498DB', lw=2, label='Train Loss (MSE)')
ax.plot(history.history['val_loss'], color='#E74C3C', lw=2, linestyle='--', label='Validation Loss (MSE)')
ax.set_title("LSTM Training & Validation Loss", fontsize=14, fontweight='bold')
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("MSE Loss", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.4)
fig.tight_layout()
savefig(fig, "fig5_lstm_loss_curve.png")

# =========================================================
# FIGURE 6 — Feature Importance (Random Forest)
# =========================================================
print("Generating Figure 6: Feature Importance...")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_feat = [feature_names[i] for i in indices]
sorted_imp  = importances[indices]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(np.arange(len(sorted_feat)), sorted_imp, color='#2ECC71', edgecolor='black', alpha=0.85)
ax.set_xticks(np.arange(len(sorted_feat)))
ax.set_xticklabels(sorted_feat, rotation=45, ha='right', fontsize=10)
ax.set_title("Feature Importances — Random Forest", fontsize=14, fontweight='bold')
ax.set_xlabel("Feature", fontsize=12)
ax.set_ylabel("Importance Score", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.35, axis='y')
ax.set_axisbelow(True)
fig.tight_layout()
savefig(fig, "fig6_feature_importance.png")

# =========================================================
# FIGURE 7 — Grand Summary Dashboard (4-panel)
# =========================================================
print("Generating Figure 7: Grand Summary Dashboard...")
fig = plt.figure(figsize=(20, 14))
fig.suptitle("NASA Li-Ion Battery Capacity Prediction — Model Summary Dashboard",
             fontsize=17, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Top row: Actual vs Predicted for each model
for col, (y_pred, label, color) in enumerate([
    (y_pred_lr,   'Linear Regression', '#E74C3C'),
    (y_pred_rf,   'Random Forest',     '#2ECC71'),
    (y_pred_lstm, 'LSTM',              '#3498DB'),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.plot(np.arange(len(y_test)), y_test[sort_idx],  color='black', lw=1.8, label='Actual')
    ax.plot(np.arange(len(y_test)), y_pred[sort_idx],  color=color,   lw=1.2, label='Predicted', alpha=0.85)
    mae_v, _, rmse_v, r2_v = metrics(y_test, y_pred)
    ax.set_title(f"{label}\nR²={r2_v:.4f}  MAE={mae_v:.4f}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Sorted Samples", fontsize=9)
    ax.set_ylabel("Capacity (Ah)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)

# Bottom left: MAE/RMSE bar
ax_bl = fig.add_subplot(gs[1, 0])
x = np.arange(len(models))
w = 0.35
ax_bl.bar(x - w/2, mae_vals,  width=w, color=COLORS, edgecolor='black', alpha=0.85, label='MAE')
ax_bl.bar(x + w/2, rmse_vals, width=w, color=COLORS, edgecolor='black', alpha=0.5, label='RMSE', hatch='//')
ax_bl.set_xticks(x); ax_bl.set_xticklabels(models, fontsize=10)
ax_bl.set_title("MAE & RMSE Comparison", fontsize=12, fontweight='bold')
ax_bl.set_ylabel("Error (Ah)", fontsize=10)
ax_bl.legend(fontsize=9)
ax_bl.grid(True, linestyle='--', alpha=0.3, axis='y')
ax_bl.set_axisbelow(True)

# Bottom middle: R² bar
ax_bm = fig.add_subplot(gs[1, 1])
ax_bm.bar(models, r2_vals, color=COLORS, edgecolor='black', alpha=0.85)
for i, v in enumerate(r2_vals):
    ax_bm.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
ax_bm.set_title("R² Score Comparison", fontsize=12, fontweight='bold')
ax_bm.set_ylabel("R²", fontsize=10)
ax_bm.set_ylim(0, 1.05)
ax_bm.grid(True, linestyle='--', alpha=0.3, axis='y')
ax_bm.set_axisbelow(True)

# Bottom right: LSTM loss curve
ax_br = fig.add_subplot(gs[1, 2])
ax_br.plot(history.history['loss'],     color='#3498DB', lw=2, label='Train Loss')
ax_br.plot(history.history['val_loss'], color='#E74C3C', lw=2, linestyle='--', label='Val Loss')
ax_br.set_title("LSTM Training Loss Curve", fontsize=12, fontweight='bold')
ax_br.set_xlabel("Epoch", fontsize=10)
ax_br.set_ylabel("MSE Loss", fontsize=10)
ax_br.legend(fontsize=9)
ax_br.grid(True, linestyle='--', alpha=0.4)

savefig(fig, "fig7_dashboard.png")

# =========================================================
# SAVE METRICS TABLE
# =========================================================
metrics_table = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'LSTM'],
    'MAE':   [mae_lr,  mae_rf,  mae_lst],
    'MSE':   [mse_lr,  mse_rf,  mse_lst],
    'RMSE':  [rmse_lr, rmse_rf, rmse_lst],
    'R2':    [r2_lr,   r2_rf,   r2_lst],
})
metrics_table.to_csv(os.path.join(REPORT_FOLDER, "final_metrics.csv"), index=False)
print("\nFinal metrics table:")
print(metrics_table.to_string(index=False))

print("\n" + "=" * 60)
print(f"All report graphs saved to: {REPORT_FOLDER}")
print("=" * 60)
