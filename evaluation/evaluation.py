# evaluation/evaluation.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import logging
import numpy as np
from config.config import Config
from preprocessing.preprocessing import DataProcessor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from scripts.train import Trainer  # Import Trainer from train.py
from models.tabtransformer import TabTransformer
from models.attention_mlp import AttentionMLP
from models.gnn_model import PatientGNN
from torch_geometric.data import Data

logger = logging.getLogger(__name__)
logger.setLevel(Config.LOG_LEVEL)

def init_model(model_type):
    if model_type == "TabTransformer":
        model = TabTransformer(
            num_features=Config.NUM_FEATURES,
            num_classes=Config.NUM_CLASSES,
            dim=Config.TRANSFORMER_DIM,
            depth=Config.TRANSFORMER_DEPTH,
            heads=Config.TRANSFORMER_HEADS
        )
    elif model_type == "AttentionMLP":
        model = AttentionMLP(
            input_dim=Config.NUM_FEATURES,
            hidden_dim=64,
            num_classes=Config.NUM_CLASSES
        )
    elif model_type == "GNN":
        model = PatientGNN(num_features=Config.NUM_FEATURES, hidden_dim=128)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return model

def compute_class_weights(y):
    # Compute class weights inversely proportional to class frequency
    unique, counts = np.unique(y, return_counts=True)
    weights = np.zeros(unique.max() + 1, dtype=np.float32)
    for cls, cnt in zip(unique, counts):
        weights[cls] = 1.0 / cnt
    # Normalize so that sum equals number of classes
    weights = weights / weights.sum() * len(unique)
    return torch.tensor(weights, dtype=torch.float32, device=Config.DEVICE)

def cross_validate_model(model_type, n_splits=5):
    processor = DataProcessor()
    features, labels = processor.load_dataset()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
    acc_list, f1_list, auc_list = [], [], []
    all_reports = []
    all_conf_matrices = []
    
    fold = 1
    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_val = features[train_idx], features[test_idx]
        y_train, y_val = labels[train_idx], labels[test_idx]
        logger.info(f"Fold {fold}: Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
        
        # Compute class weights from training split
        class_weights = compute_class_weights(y_train)
        
        # Initialize the model and trainer
        model = init_model(model_type)
        trainer = Trainer(model, Config.DEVICE, class_weights=class_weights)
        
        # For GNN, we need to convert training data into a graph
        is_graph = False
        if model_type == "GNN":
            is_graph = True
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            num_nodes = X_train_tensor.shape[0]
            edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()
            train_graph = Data(x=X_train_tensor, edge_index=edge_index, y=y_train_tensor)
            train_loader = [train_graph]  # Use a list with one graph (all training samples in one graph)
        else:
            from torch.utils.data import TensorDataset, DataLoader
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        # Train for a fixed number of epochs
        for epoch in range(Config.NUM_EPOCHS):
            loss = trainer.train_epoch(train_loader, is_graph=is_graph)
            # Optionally print per-fold loss here if desired
        # Evaluate on validation split
        if model_type == "GNN":
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            num_nodes = X_val_tensor.shape[0]
            edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()
            val_graph = Data(x=X_val_tensor, edge_index=edge_index, y=y_val_tensor).to(Config.DEVICE)
            with torch.no_grad():
                outputs = model(val_graph)
        else:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(Config.DEVICE)
            with torch.no_grad():
                outputs = model(X_val_tensor)
        
        predictions = outputs.argmax(dim=1).cpu().numpy()
        acc = np.mean(predictions == y_val)
        f1 = f1_score(y_val, predictions, zero_division=0)
        try:
            auc = roc_auc_score(y_val, torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        except Exception:
            auc = None
        
        report = classification_report(y_val, predictions, zero_division=0)
        conf_mat = confusion_matrix(y_val, predictions)
        
        logger.info(f"Fold {fold} - Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc}")
        logger.info(f"Fold {fold} Classification Report:\n{report}")
        logger.info(f"Fold {fold} Confusion Matrix:\n{conf_mat}")
        
        acc_list.append(acc)
        f1_list.append(f1)
        if auc is not None:
            auc_list.append(auc)
        all_reports.append(report)
        all_conf_matrices.append(conf_mat)
        fold += 1

    avg_acc = np.mean(acc_list)
    avg_f1 = np.mean(f1_list)
    avg_auc = np.mean(auc_list) if auc_list else None
    logger.info(f"Cross-Validation Average Accuracy: {avg_acc:.4f}")
    logger.info(f"Cross-Validation Average F1-Score: {avg_f1:.4f}")
    if avg_auc is not None:
        logger.info(f"Cross-Validation Average ROC-AUC: {avg_auc:.4f}")
    return {
        "accuracy": avg_acc,
        "f1": avg_f1,
        "roc_auc": avg_auc,
        "reports": all_reports,
        "confusion_matrices": all_conf_matrices,
    }

if __name__ == "__main__":
    # For demonstration, perform cross-validation on each model type.
    for model_name in ["TabTransformer", "GNN", "AttentionMLP"]:
        print(f"\nCross-validation for {model_name}:")
        results = cross_validate_model(model_name, n_splits=5)
        print(f"Average Accuracy: {results['accuracy']:.4f}")
        print(f"Average F1-Score: {results['f1']:.4f}")
        if results["roc_auc"] is not None:
            print(f"Average ROC-AUC: {results['roc_auc']:.4f}")
