"""
=========================================================
NASA Battery Capacity Prediction
Phase 4 : Exploratory Data Analysis (EDA)
=========================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs")
EDA_FOLDER = os.path.join(OUTPUT_FOLDER, "eda")

MODELLING_DATASET_FILE = os.path.join(OUTPUT_FOLDER, "modelling_dataset.csv")

os.makedirs(EDA_FOLDER, exist_ok=True)

print("=" * 60)
print("NASA Battery Capacity Prediction")
print("Phase 4 : Exploratory Data Analysis")
print("=" * 60)

# =========================================================
# LOAD DATASET
# =========================================================

if not os.path.exists(MODELLING_DATASET_FILE):
    raise FileNotFoundError(f"Modelling dataset not found at {MODELLING_DATASET_FILE}. Run Phase 2 first.")

df = pd.read_csv(MODELLING_DATASET_FILE)
print(f"\nLoaded modelling dataset. Shape: {df.shape}")

# Define numeric columns for analysis
numeric_cols = [
    'Cycle_Number', 'Ambient_Temperature',
    'Voltage_Mean', 'Voltage_Std', 'Voltage_Min', 'Voltage_Max',
    'Current_Mean', 'Current_Std', 'Current_Min', 'Current_Max',
    'Temperature_Mean', 'Temperature_Std', 'Temperature_Min', 'Temperature_Max',
    'Discharge_Duration', 'Re', 'Rct', 'Capacity'
]

# =========================================================
# STATISTICAL REPORTS
# =========================================================

report_lines = []
report_lines.append("=" * 60)
report_lines.append("EXPLORATORY DATA ANALYSIS REPORT")
report_lines.append("=" * 60 + "\n")

# Shape & Columns
report_lines.append(f"Dataset Shape: {df.shape}")
report_lines.append(f"Total Rows   : {len(df)}")
report_lines.append(f"Total Columns: {len(df.columns)}\n")

# Missing Values Analysis
missing_values = df.isnull().sum()
report_lines.append("--- MISSING VALUES ANALYSIS ---")
report_lines.append(missing_values.to_string() + "\n")

# Duplicate Analysis
duplicates_count = df.duplicated().sum()
report_lines.append("--- DUPLICATE ANALYSIS ---")
report_lines.append(f"Total Duplicate Rows: {duplicates_count}\n")

# Outlier Analysis using IQR
report_lines.append("--- OUTLIER ANALYSIS (IQR METHOD) ---")
outlier_summary = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = len(outliers)
    report_lines.append(f"Feature: {col:<20} | Outliers: {len(outliers):<5} ({len(outliers)/len(df)*100:.2f}%)")
report_lines.append("\n")

# Summary Statistics
report_lines.append("--- SUMMARY STATISTICS ---")
report_lines.append(df[numeric_cols].describe().transpose().to_string() + "\n")

# Save text report
report_path = os.path.join(EDA_FOLDER, "eda_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))
print(f"Saved text report: {report_path}")

# =========================================================
# CORRELATION MATRIX
# =========================================================

corr_matrix = df[numeric_cols].corr()
corr_matrix_path = os.path.join(EDA_FOLDER, "correlation_matrix.csv")
corr_matrix.to_csv(corr_matrix_path)
print(f"Saved correlation matrix CSV: {corr_matrix_path}")

# =========================================================
# PLOTS GENERATION
# =========================================================

# 1. Heatmap (pure Matplotlib)
print("\nGenerating correlation heatmap...")
fig_heat, ax_heat = plt.subplots(figsize=(14, 12), layout='constrained')
im = ax_heat.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
fig_heat.colorbar(im, ax=ax_heat, shrink=0.8, label="Correlation Coefficient")

# Show all ticks and label them with the feature names
ax_heat.set_xticks(np.arange(len(numeric_cols)))
ax_heat.set_yticks(np.arange(len(numeric_cols)))
ax_heat.set_xticklabels(numeric_cols, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
ax_heat.set_yticklabels(numeric_cols, fontsize=9)

# Loop over data dimensions and create text annotations.
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        val = corr_matrix.values[i, j]
        # Choose text color based on cell intensity for contrast
        text_color = "white" if abs(val) > 0.6 else "black"
        ax_heat.text(j, i, f"{val:.2f}",
                     ha="center", va="center", color=text_color, fontsize=8)

ax_heat.set_title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')
heat_path = os.path.join(EDA_FOLDER, "heatmap.png")
fig_heat.savefig(heat_path, dpi=300, bbox_inches='tight')
plt.close(fig_heat)
print(f"Saved heatmap: {heat_path}")

# 2. Histograms (pure Matplotlib)
print("Generating histograms grid...")
fig_hist, axes_hist = plt.subplots(6, 3, figsize=(15, 20), layout='constrained')
axes_hist = axes_hist.flatten()

for idx, col in enumerate(numeric_cols):
    ax = axes_hist[idx]
    ax.hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(f"Distribution of {col}", fontsize=10, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle='--', alpha=0.3)

hist_path = os.path.join(EDA_FOLDER, "histograms.png")
fig_hist.savefig(hist_path, dpi=300, bbox_inches='tight')
plt.close(fig_hist)
print(f"Saved histograms: {hist_path}")

# 3. Boxplots (pure Matplotlib)
print("Generating boxplots grid...")
fig_box, axes_box = plt.subplots(6, 3, figsize=(15, 20), layout='constrained')
axes_box = axes_box.flatten()

for idx, col in enumerate(numeric_cols):
    ax = axes_box[idx]
    ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightcoral', color='black'),
               medianprops=dict(color='black'),
               flierprops=dict(marker='o', markerfacecolor='red', markersize=3, linestyle='none', alpha=0.5))
    ax.set_title(f"Boxplot of {col}", fontsize=10, fontweight='bold')
    ax.set_xticklabels([])
    ax.grid(True, linestyle='--', alpha=0.3)

box_path = os.path.join(EDA_FOLDER, "boxplots.png")
fig_box.savefig(box_path, dpi=300, bbox_inches='tight')
plt.close(fig_box)
print(f"Saved boxplots: {box_path}")

print("\n" + "=" * 60)
print("Phase 4 Completed Successfully")
print("=" * 60)
