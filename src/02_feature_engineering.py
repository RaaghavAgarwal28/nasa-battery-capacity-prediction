"""
=========================================================
NASA Battery Capacity Prediction
Phase 2 : Feature Engineering
=========================================================
"""

import os
import pandas as pd
import numpy as np

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs")
EXTRACT_FOLDER = os.path.join(DATA_FOLDER, "raw_data")

CLEAN_METADATA_FILE = os.path.join(OUTPUT_FOLDER, "clean_metadata.csv")
MODELLING_DATASET_FILE = os.path.join(OUTPUT_FOLDER, "modelling_dataset.csv")

print("=" * 60)
print("NASA Battery Capacity Prediction")
print("Phase 2 : Feature Engineering")
print("=" * 60)

# =========================================================
# PROCESS DISCHARGE CYCLES
# =========================================================

if not os.path.exists(CLEAN_METADATA_FILE):
    raise FileNotFoundError(f"Clean metadata file not found at {CLEAN_METADATA_FILE}. Run Phase 1 first.")

print("\nLoading clean metadata...")
metadata = pd.read_csv(CLEAN_METADATA_FILE)
print(f"Loaded {len(metadata)} discharge cycles.")

features_list = []

print("\nProcessing time-series CSV files and extracting features...")
total_files = len(metadata)

for idx, row in metadata.iterrows():
    # Progress printing
    if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
        print(f"Processed {idx + 1} / {total_files} files...")

    file_path = os.path.join(EXTRACT_FOLDER, row['filename'])
    
    # Exception handling for missing files
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}. Skipping.")
        continue
        
    try:
        df = pd.read_csv(file_path)
        
        # Check if file is empty
        if df.empty:
            print(f"Warning: Empty file: {file_path}. Skipping.")
            continue
            
        # Extract features from the time-series CSV
        voltage = df['Voltage_measured']
        current = df['Current_measured']
        temp = df['Temperature_measured']
        time_col = df['Time']
        
        # Summary statistics
        voltage_mean = voltage.mean()
        voltage_std = voltage.std()
        voltage_min = voltage.min()
        voltage_max = voltage.max()
        
        current_mean = current.mean()
        current_std = current.std()
        current_min = current.min()
        current_max = current.max()
        
        temp_mean = temp.mean()
        temp_std = temp.std()
        temp_min = temp.min()
        temp_max = temp.max()
        
        # Discharge duration
        discharge_duration = time_col.max() - time_col.min() if not time_col.empty else 0.0
        
        # Compile row features
        features = {
            'Battery_ID': row['battery_id'],
            'Cycle_Number': int(row['Cycle_Number']),
            'Ambient_Temperature': row['ambient_temperature'],
            'Voltage_Mean': voltage_mean,
            'Voltage_Std': voltage_std,
            'Voltage_Min': voltage_min,
            'Voltage_Max': voltage_max,
            'Current_Mean': current_mean,
            'Current_Std': current_std,
            'Current_Min': current_min,
            'Current_Max': current_max,
            'Temperature_Mean': temp_mean,
            'Temperature_Std': temp_std,
            'Temperature_Min': temp_min,
            'Temperature_Max': temp_max,
            'Discharge_Duration': discharge_duration,
            'Re': float(row['Re']),
            'Rct': float(row['Rct']),
            'Capacity': float(row['Capacity'])
        }
        
        features_list.append(features)
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}. Skipping.")
        continue

# Create Modelling Dataset
modelling_df = pd.DataFrame(features_list)

# Remove NaNs
initial_shape = modelling_df.shape[0]
modelling_df = modelling_df.dropna()
final_shape = modelling_df.shape[0]

print(f"\nRemoved {initial_shape - final_shape} rows containing NaNs.")
print(f"Final dataset shape: {modelling_df.shape}")

# Save modelling dataset
modelling_df.to_csv(MODELLING_DATASET_FILE, index=False)
print(f"\nModelling dataset saved to: {MODELLING_DATASET_FILE}")

# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("Phase 2 Completed Successfully")
print("=" * 60)
