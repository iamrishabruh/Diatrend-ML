# scripts/train.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config.config import Config
from preprocessing.preprocessing import DataProcessor
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

logger = logging.getLogger(__name__)

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.LOGS_DIR / "training.log"),
            logging.StreamHandler()
        ]
    )

class Trainer:
    def __init__(self, model, device, class_weights=None):
        self.model = model.to(device)
        if class_weights is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)
        
    def train_epoch(self, loader, is_graph=False):
        self.model.train()
        total_loss = 0
        for batch in loader:
            if is_graph:
                data = batch.to(Config.DEVICE)
                outputs = self.model(data)
                labels = data.y.to(Config.DEVICE)
            else:
                inputs, labels = batch
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = self.model(inputs)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            if torch.isnan(loss):
                logger.error("Loss is NaN. Check your inputs and model outputs.")
                return float("nan")
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

def main(model_type):
    configure_logging()
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    processor = DataProcessor()
    features, labels = processor.load_dataset()
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=Config.SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=Config.SEED)

    if model_type in ["TabTransformer", "AttentionMLP"]:
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        is_graph = False
    elif model_type == "GNN":
        from models.gnn_model import PatientGNN
        model = PatientGNN(num_features=Config.NUM_FEATURES, hidden_dim=128)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        num_nodes = X_train_tensor.shape[0]
        edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()
        train_graph = Data(x=X_train_tensor, edge_index=edge_index, y=y_train_tensor)
        loader = GeoDataLoader([train_graph], batch_size=1, shuffle=True)
        is_graph = True
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model_type == "TabTransformer":
        from models.tabtransformer import TabTransformer
        model = TabTransformer(
            num_features=Config.NUM_FEATURES,
            num_classes=Config.NUM_CLASSES,
            dim=Config.TRANSFORMER_DIM,
            depth=Config.TRANSFORMER_DEPTH,
            heads=Config.TRANSFORMER_HEADS
        )
    elif model_type == "AttentionMLP":
        from models.attention_mlp import AttentionMLP
        model = AttentionMLP(
            input_dim=Config.NUM_FEATURES,
            hidden_dim=64,
            num_classes=Config.NUM_CLASSES
        )

    # Compute class weights based on y_train
    unique, counts = np.unique(y_train, return_counts=True)
    weights = np.zeros(unique.max()+1, dtype=np.float32)
    for cls, cnt in zip(unique, counts):
        weights[cls] = 1.0/cnt
    weights = weights / weights.sum() * len(unique)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=Config.DEVICE)

    trainer = Trainer(model, Config.DEVICE, class_weights=class_weights)
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = trainer.train_epoch(loader, is_graph=is_graph)
        logger.info(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Loss: {train_loss:.4f}")
        if np.isnan(train_loss):
            logger.error("Training halted due to NaN loss.")
            break
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.CHECKPOINT_PATHS[model_type])
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOP_PATIENCE:
                logger.info(f"Early stopping triggered for {model_type}")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiaTrend models")
    parser.add_argument("--model", type=str, required=True, choices=["TabTransformer", "GNN", "AttentionMLP"], help="Model architecture to train")
    args = parser.parse_args()
    main(args.model)
