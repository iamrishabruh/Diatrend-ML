import os
from pathlib import Path
import torch

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    CHECKPOINTS = BASE_DIR / "checkpoints"
    RAW_DATA = BASE_DIR / "data" / "raw"
    DEMOGRAPHICS_PATH = RAW_DATA / "demographics.xlsx"
    LOGS_DIR = BASE_DIR / "logs"
    LOGS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Common Model parameters
    NUM_FEATURES = 5  # [CGM mean, CGM std, glycemic variability, Age, Hemoglobin A1C]
    NUM_CLASSES = 2
    TRANSFORMER_DIM = 64
    TRANSFORMER_DEPTH = 4
    TRANSFORMER_HEADS = 4

    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    EARLY_STOP_PATIENCE = 10
    SEED = 42

    # GNN-specific parameters
    GNN_HIDDEN_DIM = 128
    GNN_NUM_LAYERS = 2
    GNN_DROPOUT = 0.4
    GNN_K_NEIGHBORS = 5  # Use k-nearest neighbors for graph construction

    # Synthetic augmentation settings
    AUGMENT_TRAIN = True
    AUGMENT_K_NEIGHBORS = 3

    LOG_LEVEL = "INFO"
