# File: evaluation/evaluate_all.py

import os
import glob
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import config
from preprocessing.preprocessing import load_demographics, load_and_extract
from models.tabtransformer import TabTransformer
from models.gnn_model import PatientGNN
from models.attention_mlp import AttentionMLP

def load_model(model_type, feature_dim):
    if model_type == "TabTransformer":
        model = TabTransformer(num_features=feature_dim,
                               num_classes=config.NUM_CLASSES,
                               dim=config.TRANSFORMER_DIM,
                               depth=config.TRANSFORMER_DEPTH)
        ckpt = config.CHECKPOINT_PATHS["TabTransformer"]
    elif model_type == "GNN":
        model = PatientGNN(num_features=feature_dim, hidden_dim=config.GNN_HIDDEN_DIM)
        ckpt = config.CHECKPOINT_PATHS["GNN"]
    elif model_type == "AttentionMLP":
        model = AttentionMLP(input_dim=feature_dim, hidden_dim=config.ATTENTION_HIDDEN_DIM,
                             num_classes=config.NUM_CLASSES)
        ckpt = config.CHECKPOINT_PATHS["AttentionMLP"]
    else:
        raise ValueError("Unknown model type.")
    
    model.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    return model

def main():
    # Load data
    from preprocessing.preprocessing import load_demographics
    df_demo = load_demographics(config.DEMOGRAPHICS_PATH)
    
    file_list = glob.glob(os.path.join(config.RAW_DATA_PATH, "*.xlsx"))
    X_list, y_list = [], []
    for file_path in file_list:
        try:
            feats, tgt = load_and_extract(file_path, df_demo)
            X_list.append(feats)
            y_list.append(tgt)
        except:
            pass
    
    if len(X_list) == 0:
        print("No data loaded for evaluation.")
        return
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    feature_dim = X.shape[1]
    
    # Evaluate each model
    model_types = ["TabTransformer", "GNN", "AttentionMLP"]
    for mtype in model_types:
        print(f"Evaluating {mtype}...")
        model = load_model(mtype, feature_dim)
        with torch.no_grad():
            inputs = torch.tensor(X).to(config.DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)
        report = classification_report(y, preds)
        
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
        print("-" * 40)

if __name__ == "__main__":
    main()
