"""
=========================================================
NASA Battery Capacity Prediction
Phase 1 : Data Extraction & Metadata Cleaning
=========================================================
"""

import os
import zipfile
import pandas as pd

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs")

ZIP_FILE = os.path.join(DATA_FOLDER, "data.zip")
METADATA_FILE = os.path.join(DATA_FOLDER, "metadata.csv")
EXTRACT_FOLDER = os.path.join(DATA_FOLDER, "raw_data")

# =========================================================
# CREATE FOLDERS
# =========================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

print("=" * 60)
print("NASA Battery Capacity Prediction")
print("Phase 1 : Data Extraction")
print("=" * 60)

# =========================================================
# EXTRACT ZIP
# =========================================================

print("\nExtracting data.zip ...")

# Check if extracted folder is empty or not
if not os.path.exists(EXTRACT_FOLDER) or len(os.listdir(EXTRACT_FOLDER)) == 0:
    print(f"Extracting {ZIP_FILE} to {EXTRACT_FOLDER}...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)
    print("ZIP extracted successfully.")
else:
    print("ZIP already extracted.")

# =========================================================
# LOAD AND CLEAN METADATA
# =========================================================

print("\nLoading metadata...")
metadata = pd.read_csv(METADATA_FILE)
print("Metadata loaded successfully.")
print(f"Total Rows : {len(metadata)}")

# Clean Re and Rct (convert to numeric and interpolate before filtering)
print("\nCleaning and interpolating Re and Rct values...")
metadata['Re'] = pd.to_numeric(metadata['Re'], errors='coerce')
metadata['Rct'] = pd.to_numeric(metadata['Rct'], errors='coerce')

# Interpolate within each battery_id group
metadata['Re'] = metadata.groupby('battery_id')['Re'].transform(
    lambda x: x.interpolate(method='linear').ffill().bfill()
)
metadata['Rct'] = metadata.groupby('battery_id')['Rct'].transform(
    lambda x: x.interpolate(method='linear').ffill().bfill()
)

# =========================================================
# FILTER ONLY DISCHARGE CYCLES
# =========================================================

print("\nFiltering discharge cycles...")
metadata = metadata[metadata["type"].str.lower() == "discharge"].copy()
print(f"Discharge Cycles : {len(metadata)}")

# =========================================================
# CREATE CYCLE NUMBER
# =========================================================

metadata["Cycle_Number"] = metadata.groupby("battery_id").cumcount() + 1

# =========================================================
# LOCATE AND VERIFY CSV FILES
# =========================================================

print("\nLocating CSV files recursively in raw_data...")

# Build a mapping from base filename to its relative path from EXTRACT_FOLDER
csv_mapping = {}
for root, _, files in os.walk(EXTRACT_FOLDER):
    for file in files:
        if file.endswith(".csv"):
            rel_path = os.path.relpath(os.path.join(root, file), EXTRACT_FOLDER)
            csv_mapping[file] = rel_path

# Update filenames with their correctly located paths
metadata['filename_original'] = metadata['filename']
resolved_filenames = []
missing_files = []

for idx, row in metadata.iterrows():
    orig_name = row['filename']
    # Sometimes filename in metadata is just the basename (e.g. 00001.csv)
    base_name = os.path.basename(orig_name)
    
    if base_name in csv_mapping:
        resolved_filenames.append(csv_mapping[base_name].replace('\\', '/'))
    else:
        resolved_filenames.append(orig_name)
        missing_files.append(orig_name)

metadata['filename'] = resolved_filenames

if len(missing_files) == 0:
    print("All discharge CSV files verified and located.")
else:
    print(f"WARNING: Missing Files : {len(missing_files)}")
    for f in missing_files[:10]:
        print(f)

# =========================================================
# SAVE CLEANED METADATA
# =========================================================

output_file = os.path.join(OUTPUT_FOLDER, "clean_metadata.csv")
metadata.to_csv(output_file, index=False)
print("\nClean metadata saved successfully.")
print(output_file)

# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("Phase 1 Completed Successfully")
print("=" * 60)

print(f"""
Summary
-------
Total Metadata Rows      : {len(metadata)}
Unique Batteries         : {metadata['battery_id'].nunique()}
Discharge Experiments    : {len(metadata)}
Clean Metadata Saved     : Yes
""")