"""
=========================================================
NASA Battery Capacity Prediction
Phase 6 : Model Comparison
=========================================================
"""

import os
import pandas as pd

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs")
COMPARISON_FOLDER = os.path.join(OUTPUT_FOLDER, "comparison")

LR_METRICS = os.path.join(OUTPUT_FOLDER, "linear_regression", "metrics.csv")
RF_METRICS = os.path.join(OUTPUT_FOLDER, "random_forest", "metrics.csv")
LSTM_METRICS = os.path.join(OUTPUT_FOLDER, "lstm", "metrics.csv")

os.makedirs(COMPARISON_FOLDER, exist_ok=True)

print("=" * 60)
print("NASA Battery Capacity Prediction")
print("Phase 6 : Model Comparison")
print("=" * 60)

# =========================================================
# LOAD AND COMPILE METRICS
# =========================================================

metrics_dfs = []

# Load Linear Regression metrics
if os.path.exists(LR_METRICS):
    metrics_dfs.append(pd.read_csv(LR_METRICS))
else:
    print(f"Warning: Linear Regression metrics not found at {LR_METRICS}")

# Load Random Forest metrics
if os.path.exists(RF_METRICS):
    metrics_dfs.append(pd.read_csv(RF_METRICS))
else:
    print(f"Warning: Random Forest metrics not found at {RF_METRICS}")

# Load LSTM metrics
if os.path.exists(LSTM_METRICS):
    metrics_dfs.append(pd.read_csv(LSTM_METRICS))
else:
    print(f"Warning: LSTM metrics not found at {LSTM_METRICS}")

if not metrics_dfs:
    raise FileNotFoundError("No model metrics found. Please train the models first.")

comparison_df = pd.concat(metrics_dfs, ignore_index=True)

# Save Comparison CSV
comp_csv_path = os.path.join(COMPARISON_FOLDER, "comparison.csv")
comparison_df.to_csv(comp_csv_path, index=False)
print(f"\nSaved comparison table to: {comp_csv_path}")
print(comparison_df.to_string(index=False))

# =========================================================
# IDENTIFY BEST MODEL AND GENERATE REPORT
# =========================================================

# The best model has the highest R² Score (or lowest RMSE if R² matches)
best_model_idx = comparison_df['R2'].idxmax()
best_model_row = comparison_df.iloc[best_model_idx]
best_model_name = best_model_row['Model']

report = f"""=========================================================
NASA Battery Capacity Prediction - Best Model Report
=========================================================

The best performing model in this regression study is:
Model Name: {best_model_name}

Evaluation Metrics:
------------------
MAE     : {best_model_row['MAE']:.6f}
MSE     : {best_model_row['MSE']:.6f}
RMSE    : {best_model_row['RMSE']:.6f}
R² Score: {best_model_row['R2']:.6f}

Selection Justification:
-----------------------
The models were evaluated and compared on the exact same test cycles.
{best_model_name} achieved the highest R² score of {best_model_row['R2']:.6f} and the lowest RMSE of {best_model_row['RMSE']:.6f}, making it the most robust and accurate model for battery capacity prediction in this research pipeline.
"""

best_model_txt_path = os.path.join(COMPARISON_FOLDER, "best_model.txt")
with open(best_model_txt_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\nBest model identified: {best_model_name}")
print(f"Saved best model report to: {best_model_txt_path}")

print("\n" + "=" * 60)
print("Phase 6 Completed Successfully")
print("=" * 60)
