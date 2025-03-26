# scripts/train.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RepeatedStratifiedKFold
from config.config import Config
from preprocessing.preprocessing import DataProcessor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

logger = logging.getLogger(__name__)

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOGS_DIR / 'training.log'),
            logging.StreamHandler()
        ]
    )

class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)
        
    def train_epoch(self, loader, is_graph=False):
        self.model.train()
        total_loss = 0.0
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
                return float('nan')
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader) if len(loader) > 0 else 0.0

def main(model_type, n_splits=2, n_repeats=1):
    """
    Train the specified model with repeated cross-validation.
    For each (repeat, fold) pair, we save a checkpoint:
      modelType_rep{r}_fold{f}.pt
    e.g., GNN_rep1_fold1.pt for repeat=1, fold=1.
    """
    configure_logging()
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Load entire dataset once
    processor = DataProcessor()
    features, labels = processor.load_dataset()
    features = np.array(features)
    labels = np.array(labels)

    # We use RepeatedStratifiedKFold, so each (repeat, fold) is a unique partition
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=Config.SEED
    )

    # For each repeated split => train on train_idx, no formal val set needed
    # (we do early stopping with train loss or user can adapt a manual hold-out.)
    # If you'd prefer to have an internal val set, you'd do additional splits inside this loop.
    rep_fold_idx = 1
    for fold_idx, (train_index, _) in enumerate(rskf.split(features, labels), start=1):
        # training subset
        X_train, y_train = features[train_index], labels[train_index]

        # Build the correct loader
        if model_type in ["TabTransformer","AttentionMLP"]:
            dataset_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long))
            loader_train = DataLoader(dataset_train, batch_size=Config.BATCH_SIZE, shuffle=True)
            is_graph = False

        elif model_type == "GNN":
            from models.gnn_model import PatientGNN
            X_t = torch.tensor(X_train, dtype=torch.float32)
            y_t = torch.tensor(y_train, dtype=torch.long)
            n = X_t.shape[0]
            # Build fully connected edges (excluding self loops)
            edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i!=j],
                                      dtype=torch.long).t().contiguous()
            train_graph = Data(x=X_t, edge_index=edge_index, y=y_t)
            loader_train = GeoDataLoader([train_graph], batch_size=1, shuffle=True)
            is_graph = True
        else:
            raise ValueError(f"Unsupported model: {model_type}")

        # Build model instance
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
                hidden_dim=256,
                num_classes=Config.NUM_CLASSES
            )
        elif model_type == "GNN":
            from models.gnn_model import PatientGNN
            model = PatientGNN(
                num_features=Config.NUM_FEATURES,
                hidden_dim=128
            )

        trainer = Trainer(model, Config.DEVICE)
        best_loss = float('inf')
        patience_counter = 0

        # Train loop
        for epoch in range(Config.NUM_EPOCHS):
            train_loss = trainer.train_epoch(loader_train, is_graph=is_graph)
            logger.info(f"RepeatFold {fold_idx} | Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.4f}")

            if np.isnan(train_loss):
                logger.error("Training halted due to NaN loss.")
                break
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                # Save best checkpoint => modelType_rep{repeat}_fold{fold}
                checkpoint_name = f"{model_type}_repFold{fold_idx}.pt"
                torch.save(model.state_dict(), Config.CHECKPOINTS / checkpoint_name)
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOP_PATIENCE:
                    logger.info(f"Early stopping triggered for fold {fold_idx} on {model_type}")
                    break

        rep_fold_idx += 1

if __name__ == "__main__":

    def train_model(model_type: str, folds: int, repeats: int):
        """
        Expose the existing main() logic as a callable function for Streamlit.
        """
        main(model_type, n_splits=folds, n_repeats=repeats)

    parser = argparse.ArgumentParser(description='Train DiaTrend models with repeated cross-validation')
    parser.add_argument('--model', type=str, required=True,
                        choices=["TabTransformer","GNN","AttentionMLP"],
                        help="Model architecture to train")
    parser.add_argument('--folds', type=int, default=2,
                        help="Number of folds, e.g. 2 or 3. (default=2)")
    parser.add_argument('--repeats', type=int, default=1,
                        help="Number of repeated runs, default=1")
    args = parser.parse_args()

    main(args.model, n_splits=args.folds, n_repeats=args.repeats)
