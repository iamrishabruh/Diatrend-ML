# File: config.py
import os
import torch

# Set random seed for reproducibility
SEED = 42

# Data paths
RAW_DATA_PATH = os.path.join("data", "raw")  # Folder with subject Excel files (each has CGM, Bolus)
DEMOGRAPHICS_PATH = os.path.join("data", "metadata", "demographics.xlsx")  # Excel with demographics

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

# Model parameters
NUM_CLASSES = 2  # Binary classification: Good vs. Not Good trajectory

# These might be updated once you finalize the total number of features
# For example, if CGM features + Bolus features + Demographics yield ~40 features
NUM_FEATURES = 40

# TabTransformer
TRANSFORMER_DIM = 32
TRANSFORMER_DEPTH = 4

# GNN
GNN_HIDDEN_DIM = 64

# Attention MLP
ATTENTION_HIDDEN_DIM = 64

# Checkpoint paths
CHECKPOINT_PATHS = {
    "TabTransformer": os.path.join("checkpoints", "tabtransformer_best.pt"),
    "GNN": os.path.join("checkpoints", "gnn_best.pt"),
    "AttentionMLP": os.path.join("checkpoints", "attentionmlp_best.pt"),
}

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
