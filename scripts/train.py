# File: scripts/train.py

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import glob

import config
from preprocessing.preprocessing import load_demographics, load_and_extract
from models.tabtransformer import TabTransformer
from models.gnn_model import PatientGNN
from models.attention_mlp import AttentionMLP

# Set random seeds
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)

def train_model(model, data_loader, num_epochs, checkpoint_path):
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(config.DEVICE)
            y_batch = y_batch.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
        
        # Save checkpoint if improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path} (Epoch {epoch+1}, Loss {best_loss:.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=["TabTransformer", "GNN", "AttentionMLP"],
                        help="Model type to train.")
    args = parser.parse_args()
    
    # Load demographics once
    df_demo = load_demographics(config.DEMOGRAPHICS_PATH)
    
    # Load all subject Excel files
    file_list = glob.glob(os.path.join(config.RAW_DATA_PATH, "*.xlsx"))
    X_list, y_list = [], []
    for file_path in file_list:
        try:
            feats, tgt = load_and_extract(file_path, df_demo)
            X_list.append(feats)
            y_list.append(tgt)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if len(X_list) == 0:
        print("No data loaded. Check your data files or naming conventions.")
        return
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"Loaded {X.shape[0]} subjects. Feature dim = {X.shape[1]}.")
    
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Create model
    if args.model == "TabTransformer":
        model = TabTransformer(num_features=X.shape[1],
                               num_classes=config.NUM_CLASSES,
                               dim=config.TRANSFORMER_DIM,
                               depth=config.TRANSFORMER_DEPTH)
        ckpt = config.CHECKPOINT_PATHS["TabTransformer"]
    elif args.model == "GNN":
        model = PatientGNN(num_features=X.shape[1], hidden_dim=config.GNN_HIDDEN_DIM)
        ckpt = config.CHECKPOINT_PATHS["GNN"]
        # Real GNN usage requires building a graph. 
        # For demonstration, we treat X as if it was a single batch. 
        # This might not be correct unless you implement a graph building pipeline.
    elif args.model == "AttentionMLP":
        model = AttentionMLP(input_dim=X.shape[1], hidden_dim=config.ATTENTION_HIDDEN_DIM,
                             num_classes=config.NUM_CLASSES)
        ckpt = config.CHECKPOINT_PATHS["AttentionMLP"]
    
    model.to(config.DEVICE)
    
    # Train
    train_model(model, data_loader, config.NUM_EPOCHS, ckpt)

if __name__ == "__main__":
    main()
