# File: preprocessing/preprocessing.py

import os
import re
import numpy as np
import pandas as pd
from tsfeatures import tsfeatures

###############################################################################
# 1. HELPER FUNCTIONS TO LOAD SHEETS
###############################################################################

def load_cgm_sheet(file_path: str) -> pd.DataFrame:
    """
    Load the CGM sheet from a subject's Excel file.
    Expects only columns: ['date', 'mg/dl'].
    """
    df_cgm = pd.read_excel(file_path, sheet_name="CGM")
    # Keep only the expected columns
    expected = ["date", "mg/dl"]
    missing = [col for col in expected if col not in df_cgm.columns]
    if missing:
        raise ValueError(f"Expected columns {missing} not found in CGM sheet for {file_path}")
    df_cgm = df_cgm[expected]
    # Convert 'date' to datetime
    df_cgm["date"] = pd.to_datetime(df_cgm["date"], errors="coerce")
    df_cgm = df_cgm.sort_values("date")
    return df_cgm

def load_bolus_sheet(file_path: str) -> pd.DataFrame:
    """
    Load the Bolus sheet from a subject's Excel file.
    Expects columns like 'date', 'normal', 'carbInput', etc.
    """
    df_bolus = pd.read_excel(file_path, sheet_name="Bolus")
    # Convert date to datetime
    df_bolus["date"] = pd.to_datetime(df_bolus["date"], errors="coerce")
    df_bolus = df_bolus.sort_values("date")
    return df_bolus

###############################################################################
# 2. DEMOGRAPHICS LOADER
###############################################################################

def load_demographics(demo_path: str) -> pd.DataFrame:
    """
    Load the demographics Excel file. 
    Expects columns: ['Subject', 'Age', 'Gender', 'Race', 'Hemoglobin A1C'] in Sheet1.
    """
    df_demo = pd.read_excel(demo_path, sheet_name=0)
    # Example columns: Subject, Age, Gender, Race, Hemoglobin A1C
    # You can rename or clean them if needed
    return df_demo

###############################################################################
# 3. FEATURE ENGINEERING
###############################################################################

def compute_good_trajectory(cgm_series: pd.Series) -> int:
    """
    Compute 'Good Trajectory' label:
      - >= 75% of readings between 70-180 mg/dL
      - <= 2% below 70 mg/dL
      - No consecutive hyperglycemia (>240 mg/dL) > 2 hours (24 consecutive 5-min intervals)
    """
    values = cgm_series.dropna().to_numpy()
    if len(values) == 0:
        return 0
    
    total = len(values)
    in_range = np.sum((values >= 70) & (values <= 180)) / total * 100
    below_70 = np.sum(values < 70) / total * 100

    # Check consecutive hyper
    above_240 = (values > 240).astype(int)
    max_consec = 0
    cur = 0
    for v in above_240:
        if v == 1:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0
    
    # Evaluate criteria
    if (in_range >= 75) and (below_70 <= 2) and (max_consec < 24):
        return 1
    else:
        return 0

def extract_cgm_features(cgm_series: pd.Series) -> dict:
    """
    Extract time-series features from CGM readings using tsfeatures + custom logic.
    """
    # Basic stats
    features = tsfeatures(cgm_series.dropna(), freq=5)
    
    # Glycemic variability
    diff_series = np.diff(cgm_series.dropna())
    if len(diff_series) > 1:
        features['glycemic_variability'] = np.sqrt(np.mean(diff_series**2) / (len(diff_series) - 1))
    else:
        features['glycemic_variability'] = 0.0

    # State transitions
    states = pd.cut(cgm_series, bins=[0, 70, 180, 240, 9999], labels=['hypo','norm','hyper1','hyper2'])
    transition_matrix = pd.crosstab(states[:-1], states[1:], normalize='index')
    for (row_label, col_label), val in transition_matrix.stack().to_dict().items():
        features[f"trans_{row_label}_to_{col_label}"] = val
    
    return features

def extract_bolus_features(df_bolus: pd.DataFrame) -> dict:
    """
    Example feature extraction from Bolus sheet:
      - total insulin delivered
      - average carbs
      - # of boluses
      - etc.
    """
    if df_bolus.empty:
        return {
            "total_insulin": 0.0,
            "avg_insulin": 0.0,
            "total_carbs": 0.0,
            "avg_carbs": 0.0,
            "num_boluses": 0
        }
    total_insulin = df_bolus["normal"].sum(skipna=True)
    avg_insulin = df_bolus["normal"].mean(skipna=True)
    total_carbs = df_bolus["carbInput"].sum(skipna=True)
    avg_carbs = df_bolus["carbInput"].mean(skipna=True)
    num_boluses = len(df_bolus)
    
    return {
        "total_insulin": total_insulin,
        "avg_insulin": avg_insulin,
        "total_carbs": total_carbs,
        "avg_carbs": avg_carbs,
        "num_boluses": num_boluses
    }

def extract_demographics_features(subject_id: str, df_demo: pd.DataFrame) -> dict:
    """
    Merge demographic info for a given subject ID.
    The demographics file must have a 'Subject' column that matches 'subject_id'.
    """
    row = df_demo[df_demo["Subject"] == subject_id]
    if row.empty:
        # If no match found, return empty or default
        return {
            "Age": 0,
            "Gender": "Unknown",
            "Race": "Unknown",
            "HemoglobinA1C": 0.0
        }
    else:
        row = row.iloc[0]
        # Convert categorical features to numeric or keep as is
        # E.g. map gender to {Male: 0, Female: 1} or keep as string
        # Same for Race. For demonstration, let's keep them as numeric placeholders.
        gender_map = {"Male": 0, "Female": 1}
        gender_val = gender_map.get(row["Gender"], 2)  # 2 for unknown
        race_str = str(row["Race"])  # you might do some encoding
        # For A1C, just read directly
        hba1c = float(row["Hemoglobin A1C"]) if not pd.isnull(row["Hemoglobin A1C"]) else 0.0
        
        return {
            "Age": float(row["Age"]),
            "Gender": gender_val,
            "Race": 0.0,  # placeholder for race. You can do a proper encoding if you want
            "HemoglobinA1C": hba1c
        }

###############################################################################
# 4. MAIN FUNCTION: EXTRACT ALL FEATURES + TARGET
###############################################################################

def load_and_extract(file_path: str, df_demo: pd.DataFrame) -> (list, int):
    """
    Given a subject's Excel file path, load CGM + Bolus, compute features, 
    merge with demographics, and compute target.
    
    Returns:
        (feature_vector, target_label)
    """
    # Identify subject ID from filename, e.g. "Subject21.xlsx" -> "Subject21"
    # or you might parse it differently
    basename = os.path.basename(file_path)
    match = re.match(r"(Subject\d+)\.xlsx", basename)
    if match:
        subject_id = match.group(1)
    else:
        subject_id = basename  # fallback
    
    # Load sheets
    df_cgm = load_cgm_sheet(file_path)
    df_bolus = load_bolus_sheet(file_path)
    
    # CGM-based target
    target = compute_good_trajectory(df_cgm["mg/dl"])
    
    # Extract features from CGM
    cgm_feats = extract_cgm_features(df_cgm["mg/dl"])
    
    # Extract features from Bolus
    bolus_feats = extract_bolus_features(df_bolus)
    
    # Extract demographics
    demo_feats = extract_demographics_features(subject_id, df_demo)
    
    # Merge all feature dicts
    all_feats = {}
    all_feats.update(cgm_feats)
    all_feats.update(bolus_feats)
    all_feats.update(demo_feats)
    
    # Sort by key for consistent ordering
    sorted_keys = sorted(all_feats.keys())
    feature_vector = [all_feats[k] for k in sorted_keys]
    
    return feature_vector, target
