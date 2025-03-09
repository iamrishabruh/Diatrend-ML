# File: app.py

import streamlit as st
import pandas as pd
import requests
import os
import subprocess
import torch

import config
from preprocessing.preprocessing import (
    load_demographics, load_cgm_sheet, load_bolus_sheet,
    extract_cgm_features, extract_bolus_features, compute_good_trajectory,
    extract_demographics_features
)
from utils.visualization import plot_glucose_series

API_URL = "http://127.0.0.1:8000"  # Adjust if needed

def train_all_models():
    st.info("Training all models (TabTransformer, GNN, AttentionMLP)...")
    commands = [
        "python scripts/train.py --model TabTransformer",
        "python scripts/train.py --model GNN",
        "python scripts/train.py --model AttentionMLP"
    ]
    logs = {}
    for cmd in commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            logs[cmd] = result.stdout + "\n" + result.stderr
        except Exception as e:
            logs[cmd] = str(e)
    return logs

def main():
    st.title("DiaTrend Trajectory Predictor Dashboard")
    st.markdown("""
    **Upload a DiaTrend Excel file** that has 'CGM' and 'Bolus' sheets.
    We'll extract CGM data from 'mg/dl' column, Bolus data from columns like 'normal', 'carbInput', etc.,
    and optionally merge demographics if the subject ID matches our metadata file.
    """)

    # Upload file
    uploaded_file = st.file_uploader("Upload Subject Excel File", type=["xlsx"])
    if uploaded_file is not None:
        # Try to read CGM and Bolus
        try:
            df_cgm = pd.read_excel(uploaded_file, sheet_name="CGM")
            df_bolus = pd.read_excel(uploaded_file, sheet_name="Bolus")
            st.success("Loaded CGM & Bolus sheets!")
            st.write("CGM head:")
            st.dataframe(df_cgm.head())
            st.write("Bolus head:")
            st.dataframe(df_bolus.head())

            # Visualize CGM
            if "mg/dl" in df_cgm.columns:
                fig = plot_glucose_series(df_cgm["mg/dl"], title="CGM Glucose Readings")
                st.plotly_chart(fig)

                # Compute features
                cgm_feats = extract_cgm_features(df_cgm["mg/dl"])
                bolus_feats = extract_bolus_features(df_bolus)
                
                # Attempt demographics
                df_demo = load_demographics(config.DEMOGRAPHICS_PATH)
                # For demonstration, parse subject ID from filename
                filename = os.path.basename(uploaded_file.name)
                subject_id = filename.replace(".xlsx", "")  # e.g. "Subject21"
                demo_feats = extract_demographics_features(subject_id, df_demo)

                # Merge all
                all_feats = {}
                all_feats.update(cgm_feats)
                all_feats.update(bolus_feats)
                all_feats.update(demo_feats)

                # Compute target
                tgt = compute_good_trajectory(df_cgm["mg/dl"])
                
                st.markdown("### Extracted Features")
                st.write(all_feats)
                st.markdown(f"**Computed Target (Good Trajectory)**: {tgt}")

            else:
                st.error("No 'mg/dl' column found in CGM sheet!")
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")

    st.markdown("---")
    st.header("Model Training & Evaluation")
    if st.button("Train All Models"):
        logs = train_all_models()
        for cmd, log in logs.items():
            st.text(f"{cmd}\n{log}\n")
    
    if st.button("Evaluate Models"):
        st.info("Running 'python evaluation/evaluate_all.py'... Check terminal output/logs for results.")
        subprocess.run("python evaluation/evaluate_all.py", shell=True)

    st.markdown("---")
    st.header("Query API for Predictions")
    if st.button("Predict from This File via API"):
        if uploaded_file is not None:
            # We'll do a /predict_excel call
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            try:
                r = requests.post(f"{API_URL}/predict_excel", files=files)
                if r.ok:
                    st.success(f"API Prediction: {r.json()}")
                else:
                    st.error(f"API error: {r.text}")
            except Exception as e:
                st.error(f"Error contacting API: {e}")
        else:
            st.error("No file uploaded.")

if __name__ == "__main__":
    main()
