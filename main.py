"""
=========================================================
NASA Battery Capacity Predictor
Main Pipeline Orchestrator
Author: Raaghav Agarwal
=========================================================
"""

import os
import subprocess
import sys
import time

def run_script(script_path):
    """Runs a python script as a subprocess and streams its output."""
    print("=" * 70)
    print(f"RUNNING: {script_path}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run the script and inherit the current stdout/stderr streams
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    if result.returncode == 0:
        print(f"\nSUCCESS: {script_path} completed in {elapsed:.2f} seconds.\n")
        return True
    else:
        print(f"\nFAILURE: {script_path} exited with code {result.returncode} after {elapsed:.2f} seconds.\n")
        return False

def main():
    pipeline_start = time.time()
    
    # Define the execution pipeline sequence
    scripts = [
        "src/01_extract_data.py",
        "src/02_feature_engineering.py",
        "src/03_engineering_plots.py",
        "src/04_eda.py",
        "src/05_linear_regression.py",
        "src/06_random_forest.py",
        "src/07_lstm.py",
        "src/08_compare_models.py"
    ]
    
    print("*" * 70)
    print("NASA LITHIUM-ION BATTERY AGEING CAPACITY PREDICTOR PIPELINE")
    print("*" * 70)
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Python Executable       : {sys.executable}")
    print(f"Total steps to run      : {len(scripts)}")
    print("*" * 70 + "\n")
    
    for idx, script in enumerate(scripts, 1):
        script_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
        if not os.path.exists(script_full_path):
            print(f"CRITICAL ERROR: Script not found: {script_full_path}")
            sys.exit(1)
            
        print(f"[STEP {idx}/{len(scripts)}] Starting...")
        success = run_script(script_full_path)
        
        if not success:
            print(f"Pipeline execution aborted at step {idx}: {script}")
            sys.exit(1)
            
    pipeline_elapsed = time.time() - pipeline_start
    print("*" * 70)
    print("NASA BATTERY PREDICTOR PIPELINE EXECUTED SUCCESSFULLY")
    print(f"Total elapsed time: {pipeline_elapsed / 60.0:.2f} minutes")
    print("*" * 70)

if __name__ == "__main__":
    main()
