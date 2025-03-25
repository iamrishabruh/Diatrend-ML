# config/config.py
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

    # Model parameters
    NUM_FEATURES = 5  # [CGM mean, CGM std, glycemic variability, Age, Hemoglobin A1C]
    NUM_CLASSES = 2
    TRANSFORMER_DIM = 128
    TRANSFORMER_DEPTH = 8
    TRANSFORMER_HEADS = 8

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-2
    EARLY_STOP_PATIENCE = 5
    SEED = 42

    CHECKPOINT_PATHS = {
        "TabTransformer": CHECKPOINTS / "TabTransformer_best.pt",
        "GNN": CHECKPOINTS / "GNN_best.pt",
        "AttentionMLP": CHECKPOINTS / "AttentionMLP_best.pt"
    }

    LOG_LEVEL = "INFO"
