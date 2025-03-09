# File: api/api_server.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
import pandas as pd
import io
import os

import config
from models.tabtransformer import TabTransformer
from preprocessing.preprocessing import load_cgm_sheet, extract_cgm_features, compute_good_trajectory

app = FastAPI(title="DiaTrend Trajectory Prediction API")

# Load the TabTransformer checkpoint by default
model = TabTransformer(num_features=config.NUM_FEATURES,
                       num_classes=config.NUM_CLASSES,
                       dim=config.TRANSFORMER_DIM,
                       depth=config.TRANSFORMER_DEPTH)
checkpoint_path = config.CHECKPOINT_PATHS["TabTransformer"]
try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model checkpoint: {e}")

class PredictRequest(BaseModel):
    features: dict  # a dictionary of {feature_name: value}

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict from a dictionary of features. 
    The code expects sorted feature keys to match training order.
    """
    sorted_keys = sorted(request.features.keys())
    vector = [request.features[k] for k in sorted_keys]
    x = torch.tensor(vector).float().unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        output = model(x)
    pred = output.argmax(dim=1).item()
    return {"prediction": pred}

@app.post("/predict_excel")
def predict_excel(file: UploadFile = File(...)):
    """
    Predict from an Excel file that has a 'CGM' sheet with 'mg/dl' column.
    We'll only use CGM features here for demonstration.
    """
    try:
        contents = file.file.read()
        df_cgm = pd.read_excel(io.BytesIO(contents), sheet_name="CGM")
        # Extract CGM features
        cgm_feats = extract_cgm_features(df_cgm["mg/dl"])
        sorted_keys = sorted(cgm_feats.keys())
        vector = [cgm_feats[k] for k in sorted_keys]
        x = torch.tensor(vector).float().unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            output = model(x)
        pred = output.argmax(dim=1).item()
        return {"prediction": pred}
    except Exception as e:
        return {"error": str(e)}
