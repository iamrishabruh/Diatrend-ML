# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from io import BytesIO
from config.config import Config
from preprocessing.preprocessing import DataProcessor
from torch_geometric.data import Data
from models.tabtransformer import TabTransformer
from models.attention_mlp import AttentionMLP
from models.gnn_model import PatientGNN
import os

##########################################
# NEW IMPORTS FOR FASTAPI
##########################################
from fastapi import FastAPI
from typing import Dict

# Create a FastAPI instance so it won't 404 when run via uvicorn
fastapi_app = FastAPI()

@fastapi_app.get("/")
def read_root() -> Dict[str, str]:
    return {
        "message": "Hello! You’ve reached the DiaTrend API root using FastAPI. "
                   "If you want the Streamlit UI, run 'streamlit run app.py'."
    }

##########################################
# STREAMLIT CODE STARTS BELOW
##########################################

def build_single_model(model_type):
    if model_type == "TabTransformer":
        return TabTransformer(
            num_features=Config.NUM_FEATURES,
            num_classes=Config.NUM_CLASSES,
            dim=Config.TRANSFORMER_DIM,
            depth=Config.TRANSFORMER_DEPTH,
            heads=Config.TRANSFORMER_HEADS
        )
    elif model_type == "AttentionMLP":
        return AttentionMLP(
            input_dim=Config.NUM_FEATURES,
            hidden_dim=256,  # or 64
            num_classes=Config.NUM_CLASSES
        )
    elif model_type == "GNN":
        return PatientGNN(
            num_features=Config.NUM_FEATURES,
            hidden_dim=128
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def load_fold_checkpoint(model_type, fold_index):
    """
    Loads a single model checkpoint from e.g. TabTransformer_repFold{fold_index}.pt
    """
    ckpt_name = f"{model_type}_repFold{fold_index}.pt"
    ckpt_path = Config.CHECKPOINTS / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_name} not found in {Config.CHECKPOINTS}")
    model_obj = build_single_model(model_type)
    state = torch.load(ckpt_path, map_location=Config.DEVICE)
    model_obj.load_state_dict(state)
    model_obj.to(Config.DEVICE)
    model_obj.eval()
    return model_obj

def single_sample_inference(model_obj, features, model_type):
    """
    Inference for single sample if using one model.
    model_type used to decide if we pass data.x or the entire Data object for GNN.
    """
    import torch
    from torch_geometric.data import Data

    # shape [1, num_features]
    x_tensor = torch.tensor([features], dtype=torch.float32).to(Config.DEVICE)

    if model_type == "GNN":
        # single-node graph with a self-loop
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        data = Data(x=x_tensor, edge_index=edge_index).to(Config.DEVICE)
        with torch.no_grad():
            logits = model_obj(data)  # shape [1,2]
        pred = int(logits.argmax(dim=1).item())
    else:
        with torch.no_grad():
            logits = model_obj(x_tensor)  # shape [1,2]
        pred = int(logits.argmax(dim=1).item())
    return pred

def ensemble_single_sample_inference(features, fold_index):
    """
    Load each model's repeated-fold checkpoint, average logits, do single-sample inference.
    """
    import torch
    from torch_geometric.data import Data

    # Build each single model
    model_tt = load_fold_checkpoint("TabTransformer", fold_index)
    model_mlp = load_fold_checkpoint("AttentionMLP", fold_index)
    model_gnn = load_fold_checkpoint("GNN", fold_index)

    x_tensor = torch.tensor([features], dtype=torch.float32).to(Config.DEVICE)
    # GNN approach
    edge_index = torch.tensor([[0],[0]], dtype=torch.long)
    data = Data(x=x_tensor, edge_index=edge_index).to(Config.DEVICE)

    with torch.no_grad():
        # 1) TabTransformer => pass x_tensor
        logits_tt = model_tt(x_tensor)
        # 2) MLP => pass x_tensor
        logits_mlp = model_mlp(x_tensor)
        # 3) GNN => pass entire data
        logits_gnn = model_gnn(data)

        avg_logits = (logits_tt + logits_mlp + logits_gnn) / 3.0
        pred = int(avg_logits.argmax(dim=1).item())
    return pred

def predict_model(model_choice, fold_index, features):
    """
    Single-sample inference with the chosen model or ensemble, specifying which fold's checkpoint to load.
    """
    if model_choice in ["TabTransformer","AttentionMLP","GNN"]:
        # Single model from fold
        model_obj = load_fold_checkpoint(model_choice, fold_index)
        prediction = single_sample_inference(model_obj, features, model_choice)
    elif model_choice == "Ensemble":
        # Merge logits from all 3
        prediction = ensemble_single_sample_inference(features, fold_index)
    else:
        prediction = None
    return prediction

##########################################
# The Streamlit "main" for local app use
##########################################
def main():
    st.set_page_config(page_title="DiaTrend Dashboard", layout="wide")
    tabs = st.tabs(["Train","Evaluate","Inference","Manage Checkpoints"])

    with tabs[0]:
        st.header("Train Model")
        model = st.selectbox("Model", ["TabTransformer","AttentionMLP","GNN"])
        folds = st.number_input("Folds", min_value=2, max_value=10, value=2)
        repeats = st.number_input("Repeats", min_value=1, max_value=5, value=1)
        if st.button("Train"):
            from scripts.train import train_model
            train_model(model, folds, repeats)
            st.success("Training complete!")

    with tabs[1]:
        st.header("Evaluate")
        model = st.selectbox("Eval Mode", ["TabTransformer","AttentionMLP","GNN","Ensemble"])
        folds = st.number_input("Eval Folds", 2, 10, key="eval_folds")
        repeats = st.number_input("Eval Repeats", 1, 5, key="eval_repeats")
        if st.button("Evaluate"):
            from evaluation.evaluation import evaluate_all
            folds_metrics, overall = evaluate_all(model, folds, repeats)
            df = pd.DataFrame(folds_metrics, columns=["Loss","Accuracy","AUC","Sensitivity","Specificity"])
            st.dataframe(df)
            st.json(overall)

    with tabs[2]:
        st.header("Single Inference")
        mode = st.selectbox("Mode", ["TabTransformer","AttentionMLP","GNN","Ensemble"])
        fold_idx = st.number_input("Fold Index", min_value=1, value=1)
        file = st.file_uploader("Upload Subject XLSX", type="xlsx")
        if file and st.button("Predict"):
            dp = DataProcessor(); demo = dp._load_demographics()
            feats, _ = dp.process_file(file, demo, filename=file.name)
            from app import predict_model
            pred = predict_model(mode, fold_idx, feats)
            st.metric("Prediction", "Good Control" if pred==1 else "Needs Intervention")

    with tabs[3]:
        st.header("Manage Checkpoints")
        if st.button("Delete All Checkpoints"):
            import shutil
            shutil.rmtree(Config.CHECKPOINTS, ignore_errors=True)
            Config.CHECKPOINTS.mkdir(exist_ok=True)
            st.success("All checkpoints removed.")


##########################################
# Standard Python entry point
##########################################
if __name__ == "__main__":
    # If you run:  python app.py
    # this will invoke the Streamlit UI (through `streamlit run app.py`)
    # unless you explicitly run "uvicorn app:fastapi_app --reload"
    main()
