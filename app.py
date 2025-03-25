import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from io import BytesIO
from config.config import Config
from preprocessing.preprocessing import DataProcessor
from models.ensemble import ensemble_predict
from models.tabtransformer import TabTransformer
from models.attention_mlp import AttentionMLP
from models.gnn_model import PatientGNN
from torch_geometric.data import Data

def load_tabtransformer():
    model = TabTransformer(
        num_features=Config.NUM_FEATURES,
        num_classes=Config.NUM_CLASSES,
        dim=Config.TRANSFORMER_DIM,
        depth=Config.TRANSFORMER_DEPTH,
        heads=Config.TRANSFORMER_HEADS
    )
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATHS["TabTransformer"], map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

def load_attention_mlp():
    model = AttentionMLP(
        input_dim=Config.NUM_FEATURES,
        hidden_dim=64,
        num_classes=Config.NUM_CLASSES
    )
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATHS["AttentionMLP"], map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

def load_gnn():
    model = PatientGNN(
        num_features=Config.NUM_FEATURES,
        hidden_dim=128
    )
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATHS["GNN"], map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

def predict_model(model_choice, features):
    tensor_input = torch.tensor([features], dtype=torch.float32).to(Config.DEVICE)

    if model_choice == "TabTransformer":
        model = load_tabtransformer()
        with torch.no_grad():
            output = model(tensor_input)
        prediction = int(output.argmax(dim=1).item())

    elif model_choice == "AttentionMLP":
        model = load_attention_mlp()
        with torch.no_grad():
            output = model(tensor_input)
        prediction = int(output.argmax(dim=1).item())

    elif model_choice == "GNN":
        x = torch.tensor([features], dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index).to(Config.DEVICE)
        model = load_gnn()
        with torch.no_grad():
            output = model(graph)
            output = output[0].unsqueeze(0)
        prediction = int(output.argmax(dim=1).item())

    elif model_choice == "Ensemble":
        prediction = ensemble_predict(features)

    else:
        prediction = None

    return prediction

def main():
    st.set_page_config(page_title="DiaTrend Analyzer", layout="wide")
    st.title("Diabetes Trajectory Analysis")

    model_choice = st.selectbox("Select Prediction Mode", ["TabTransformer", "AttentionMLP", "GNN", "Ensemble"])

    uploaded_file = st.file_uploader("Upload subject data (XLSX)", type="xlsx")

    if uploaded_file:
        try:
            processor = DataProcessor()
            demo_df = processor._load_demographics()

            # Convert UploadedFile to BytesIO for Pandas
            file_buffer = BytesIO(uploaded_file.getvalue())

            # Extract subject ID correctly
            filename = uploaded_file.name

            # Process file from in-memory buffer
            features, _ = processor.process_file(file_buffer, demo_df, filename=filename)

            with st.expander("Raw Data Preview"):
                cgm_df = pd.read_excel(file_buffer, sheet_name="CGM")
                st.dataframe(cgm_df.head())
                fig = px.line(cgm_df, x=cgm_df.index, y='mg/dl', title="Continuous Glucose Monitoring")
                st.plotly_chart(fig)

            feature_names = [
                "Average Glucose Level",
                "Glucose Variability (Standard Deviation)",
                "Glycemic Variability",
                "Age",
                "Hemoglobin A1C"
            ]

            feature_dict = {feature_names[i]: val for i, val in enumerate(features)}

            st.subheader("Processed Features")
            st.json(feature_dict)

            prediction = predict_model(model_choice, features)

            st.metric("Predicted Trajectory",
                      value="Positive" if prediction == 1 else "Needs Intervention",
                      delta="Good Control" if prediction == 1 else "High Risk")

        except Exception as e:
            st.error(f"Processing Error: {str(e)}")

if __name__ == "__main__":
    main()
