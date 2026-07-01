"""
=========================================================
NASA Battery Capacity Prediction
Helper Utilities
=========================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(filepath, seq_length=5, test_size=0.2, random_state=42):
    """
    Loads modelling_dataset.csv, builds sequences grouped by battery,
    and returns scaled datasets for flat and sequential models.
    """
        # Load dataset
    df = pd.read_csv(filepath)
    # Filter out unrealistic Re and Rct values (keep within (0,5))
    df = df[(df['Re'] > 0) & (df['Re'] < 5) & (df['Rct'] > 0) & (df['Rct'] < 5)]
    
    # Feature columns (16 features)
    feature_cols = [
        'Cycle_Number', 'Ambient_Temperature',
        'Voltage_Mean', 'Voltage_Std', 'Voltage_Min', 'Voltage_Max',
        'Current_Mean', 'Current_Std', 'Current_Min', 'Current_Max',
        'Temperature_Mean', 'Temperature_Std', 'Temperature_Min', 'Temperature_Max',
        'Discharge_Duration', 'Re', 'Rct'
    ]
    
    target_col = 'Capacity'
    
    # Build sequences within each battery
    X_flat_list = [] # current cycle features for LR and RF
    X_seq_list = []  # sequence features for LSTM
    y_list = []      # capacity targets
    
    for battery_id, group in df.groupby('Battery_ID'):
        group = group.sort_values('Cycle_Number')
        features = group[feature_cols].values
        targets = group[target_col].values
        
        # If battery doesn't have enough cycles to build sequence, skip
        if len(group) < seq_length:
            continue
            
        for i in range(seq_length - 1, len(group)):
            # Sequence from i - seq_length + 1 to i
            seq = features[i - seq_length + 1 : i + 1]
            # Current cycle features at i
            flat = features[i]
            # Target capacity at i
            target = targets[i]
            
            X_seq_list.append(seq)
            X_flat_list.append(flat)
            y_list.append(target)
            
    X_seq = np.array(X_seq_list)
    X_flat = np.array(X_flat_list)
    y = np.array(y_list)
    
    # Perform train-test split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    
    # We will scale features using StandardScaler
    
    # 1. Scale Flat features (for LR and RF)
    scaler_flat = StandardScaler()
    X_train_flat = scaler_flat.fit_transform(X_flat[train_idx])
    X_test_flat = scaler_flat.transform(X_flat[test_idx])
    
    # 2. Scale Sequence features (for LSTM)
    N_train, W, F = X_seq[train_idx].shape
    N_test = len(test_idx)
    
    X_train_seq_reshaped = X_seq[train_idx].reshape(-1, F)
    scaler_seq = StandardScaler()
    scaler_seq.fit(X_train_seq_reshaped)
    
    X_train_seq = scaler_seq.transform(X_train_seq_reshaped).reshape(N_train, W, F)
    X_test_seq = scaler_seq.transform(X_seq[test_idx].reshape(-1, F)).reshape(N_test, W, F)
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    return {
        'X_train_flat': X_train_flat,
        'X_test_flat': X_test_flat,
        'X_train_seq': X_train_seq,
        'X_test_seq': X_test_seq,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols
    }
