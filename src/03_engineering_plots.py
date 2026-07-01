"""
=========================================================
NASA Battery Capacity Prediction
Phase 3 : Engineering Plots
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

DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs")
EXTRACT_FOLDER = os.path.join(DATA_FOLDER, "raw_data")
PLOT_FOLDER = os.path.join(OUTPUT_FOLDER, "engineering_plots")

CLEAN_METADATA_FILE = os.path.join(OUTPUT_FOLDER, "clean_metadata.csv")

os.makedirs(PLOT_FOLDER, exist_ok=True)

print("=" * 60)
print("NASA Battery Capacity Prediction")
print("Phase 3 : Engineering Plots")
print("=" * 60)

# =========================================================
# LOAD METADATA
# =========================================================

if not os.path.exists(CLEAN_METADATA_FILE):
    raise FileNotFoundError(f"Clean metadata file not found at {CLEAN_METADATA_FILE}. Run Phase 1 first.")

metadata = pd.read_csv(CLEAN_METADATA_FILE)
batteries = sorted(metadata['battery_id'].unique())
print(f"Loaded clean metadata. Total batteries: {len(batteries)}")

# Set up distinct colors for each battery using a colormap
colors = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, len(batteries)))
battery_colors = dict(zip(batteries, colors))

# Sample step for plotting: we plot every 20th cycle per battery
# to show degradation over time while maintaining readability and rendering speed
SAMPLE_STEP = 20

# Initialize figures
fig_vol, ax_vol = plt.subplots(figsize=(10, 6), layout='constrained')
fig_cur, ax_cur = plt.subplots(figsize=(10, 6), layout='constrained')
fig_tmp, ax_tmp = plt.subplots(figsize=(10, 6), layout='constrained')

# Track which batteries have labels in the legend to avoid duplicates
labeled_batteries = set()

print("\nReading data and plotting profiles (sampling every 20th cycle)...")

for battery in batteries:
    bat_df = metadata[metadata['battery_id'] == battery].sort_values('Cycle_Number')
    sampled_bat_df = bat_df.iloc[::SAMPLE_STEP]
    
    color = battery_colors[battery]
    
    for _, row in sampled_bat_df.iterrows():
        file_path = os.path.join(EXTRACT_FOLDER, row['filename'])
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path)
            if df.empty or 'Time' not in df.columns:
                continue
                
            label = battery if battery not in labeled_batteries else None
            labeled_batteries.add(battery)
            
            # 1. Voltage vs Time
            if 'Voltage_measured' in df.columns:
                ax_vol.plot(df['Time'], df['Voltage_measured'], color=color, alpha=0.6, linewidth=1.0, label=label)
                
            # 2. Current vs Time
            if 'Current_measured' in df.columns:
                ax_cur.plot(df['Time'], df['Current_measured'], color=color, alpha=0.6, linewidth=1.0, label=label)
                
            # 3. Temperature vs Time
            if 'Temperature_measured' in df.columns:
                ax_tmp.plot(df['Time'], df['Temperature_measured'], color=color, alpha=0.6, linewidth=1.0, label=label)
                
        except Exception as e:
            # Silently skip file read issues here to keep output clean
            continue

print("Finalizing plots and saving figures...")

# 1. Finalize Voltage Plot
ax_vol.set_title("Discharge Voltage Profile over Lifetime (All Batteries)", fontsize=14, fontweight='bold')
ax_vol.set_xlabel("Time (seconds)", fontsize=12)
ax_vol.set_ylabel("Voltage (V)", fontsize=12)
ax_vol.grid(True, linestyle='--', alpha=0.5)
ax_vol.legend(title="Battery ID", bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=8)
vol_path = os.path.join(PLOT_FOLDER, "voltage_all_batteries.png")
fig_vol.savefig(vol_path, dpi=300, bbox_inches='tight')
print(f"Saved: {vol_path}")
plt.close(fig_vol)

# 2. Finalize Current Plot
ax_cur.set_title("Discharge Current Profile over Lifetime (All Batteries)", fontsize=14, fontweight='bold')
ax_cur.set_xlabel("Time (seconds)", fontsize=12)
ax_cur.set_ylabel("Current (A)", fontsize=12)
ax_cur.grid(True, linestyle='--', alpha=0.5)
ax_cur.legend(title="Battery ID", bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=8)
cur_path = os.path.join(PLOT_FOLDER, "current_all_batteries.png")
fig_cur.savefig(cur_path, dpi=300, bbox_inches='tight')
print(f"Saved: {cur_path}")
plt.close(fig_cur)

# 3. Finalize Temperature Plot
ax_tmp.set_title("Discharge Temperature Profile over Lifetime (All Batteries)", fontsize=14, fontweight='bold')
ax_tmp.set_xlabel("Time (seconds)", fontsize=12)
ax_tmp.set_ylabel("Temperature (°C)", fontsize=12)
ax_tmp.grid(True, linestyle='--', alpha=0.5)
ax_tmp.legend(title="Battery ID", bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=8)
tmp_path = os.path.join(PLOT_FOLDER, "temperature_all_batteries.png")
fig_tmp.savefig(tmp_path, dpi=300, bbox_inches='tight')
print(f"Saved: {tmp_path}")
plt.close(fig_tmp)

print("\n" + "=" * 60)
print("Phase 3 Completed Successfully")
print("=" * 60)
