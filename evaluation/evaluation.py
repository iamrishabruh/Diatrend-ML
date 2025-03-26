# evaluation/evaluation.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from config.config import Config
from preprocessing.preprocessing import DataProcessor

logger = logging.getLogger(__name__)
logger.setLevel(Config.LOG_LEVEL)

############################################
# SINGLE MODEL LOADER
############################################
def build_single_model(model_type):
    """
    Instantiates a single model object with the correct config but does not load any checkpoint.
    """
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
            hidden_dim=256,  # or 64, depending on your final config
            num_classes=Config.NUM_CLASSES
        )
    elif model_type == "GNN":
        from models.gnn_model import PatientGNN
        model = PatientGNN(num_features=Config.NUM_FEATURES, hidden_dim=128)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return model

def load_checkpoint(model, checkpoint_name):
    """
    Loads the state_dict from 'checkpoint_name' into 'model'.
    Returns the model in eval mode, placed on Config.DEVICE.
    """
    checkpoint_path = Config.CHECKPOINTS / checkpoint_name
    state = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.to(Config.DEVICE)
    model.eval()
    return model

############################################
# SINGLE MODEL EVALUATION
############################################
def evaluate_single_model(model, loader, is_graph=False):
    """
    Evaluate a single model (already loaded with checkpoint) on a given test loader.
    Returns (loss, accuracy, auc, sensitivity, specificity).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if is_graph:
                data = batch.to(Config.DEVICE)
                outputs = model(data)            # GNN expects 'data'
                labels = data.y
            else:
                inputs, labels = batch          # [batch_size, num_features], [batch_size]
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader)>0 else 0.0
    accuracy = correct / total if total>0 else 0.0

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, (np.array(all_probs)>=0.5).astype(int))
    if cm.shape == (2,2):
        TN, FP, FN, TP = cm.ravel()
        sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
        spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
    else:
        sens, spec = 0.0, 0.0

    return avg_loss, accuracy, auc, sens, spec

############################################
# ENSEMBLE EVALUATION (AVERAGING LOGITS)
############################################
def build_ensemble_for_fold(fold_count):
    """
    For a given fold index (fold_count), load TabTransformer, GNN, and AttentionMLP
    from fold-specific checkpoints, e.g. TabTransformer_repFold{fold_count}.pt
    """
    # We'll build each single model object, then load the matching checkpoint.
    models = {}
    for mtype in ["TabTransformer","AttentionMLP","GNN"]:
        model_obj = build_single_model(mtype)
        ckpt_name = f"{mtype}_repFold{fold_count}.pt"
        loaded = load_checkpoint(model_obj, ckpt_name)
        models[mtype] = loaded
    return models

def evaluate_ensemble(models_dict, loader):
    """
    Evaluate the ensemble on the given loader by merging each batch's logits from
    TabTransformer (which expects data.x), MLP (which expects data.x),
    GNN (which uses the entire data object).
    We compute cross-entropy on the averaged logits, plus advanced metrics.
    Returns (loss, acc, auc, sens, spec).
    """
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            data = batch.to(Config.DEVICE)       # data is a PyG Data object
            labels = data.y

            # x => shape [N, num_features]
            # GNN needs the entire 'data', while TabTransformer/MLP expect x.
            # However, in a single-batch PyG scenario, data.x => shape [N, 5].
            x = data.x                            # We pass 'x' to TT and MLP

            # 1) TabTransformer
            out_tt = models_dict["TabTransformer"](x)

            # 2) AttentionMLP
            out_mlp = models_dict["AttentionMLP"](x)

            # 3) GNN
            out_gnn = models_dict["GNN"](data)

            # average the 3 sets of logits
            avg_logits = (out_tt + out_mlp + out_gnn) / 3.0

            # cross-entropy loss
            loss = criterion(avg_logits, labels)
            total_loss += loss.item()

            # predictions
            probs = torch.softmax(avg_logits, dim=1)
            preds = avg_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader)>0 else 0.0
    accuracy = correct / total if total>0 else 0.0

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, (np.array(all_probs)>=0.5).astype(int))
    if cm.shape == (2,2):
        TN, FP, FN, TP = cm.ravel()
        sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
        spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
    else:
        sens, spec = 0.0, 0.0

    return avg_loss, accuracy, auc, sens, spec

def evaluate_main(model_type: str, n_splits: int, n_repeats: int):
    """
    Same logic as main(), but returns per-fold metrics and an overall summary.
    """
    processor = DataProcessor()
    X_all, y_all = processor.load_dataset()
    X_all, y_all = np.array(X_all), np.array(y_all)

    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=Config.SEED)

    fold_metrics = []
    fold_count = 1

    for _, test_idx in rskf.split(X_all, y_all):
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        # build loader & evaluate
        if model_type in ["TabTransformer","AttentionMLP","GNN"]:
            dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                    torch.tensor(y_test, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
            is_graph = (model_type == "GNN")
            model = build_single_model(model_type)
            load_checkpoint(model, f"{model_type}_repFold{fold_count}.pt")
            metrics = evaluate_single_model(model, loader, is_graph=is_graph)
        else:
            # ensemble
            X_t = torch.tensor(X_test, dtype=torch.float32)
            y_t = torch.tensor(y_test, dtype=torch.long)
            n = X_t.shape[0]
            edge_index = torch.tensor([[i,j] for i in range(n) for j in range(n) if i!=j], dtype=torch.long).t().contiguous()
            loader = GeoDataLoader([Data(x=X_t, edge_index=edge_index, y=y_t)], batch_size=1, shuffle=False)
            models_dict = build_ensemble_for_fold(fold_count)
            metrics = evaluate_ensemble(models_dict, loader)

        fold_metrics.append(metrics)
        fold_count += 1

    # compute averages
    avg = tuple(np.mean([m[i] for m in fold_metrics]) for i in range(5))
    return fold_metrics, avg

############################################
# MAIN: RepeatedStratifiedKFold for Single or Ensemble
############################################
def main(model_type, n_splits=2, n_repeats=1):
    """
    If model_type in {TabTransformer, GNN, AttentionMLP}, do single-model repeated CV evaluation.
    If model_type == 'Ensemble', load each fold's 3 checkpoints (TT, MLP, GNN) and average their logits.
    """
    processor = DataProcessor()
    X_all, y_all = processor.load_dataset()
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=Config.SEED
    )

    fold_losses = []
    fold_accuracies = []
    fold_aucs = []
    fold_sensitivities = []
    fold_specificities = []

    fold_count = 1
    for train_idx, test_idx in rskf.split(X_all, y_all):
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        if model_type in ["TabTransformer","AttentionMLP","GNN"]:
            # Single model approach => build test loader
            if model_type in ["TabTransformer","AttentionMLP"]:
                dataset_test = TensorDataset(
                    torch.tensor(X_test, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.long)
                )
                loader_test = DataLoader(dataset_test, batch_size=Config.BATCH_SIZE, shuffle=False)
                is_graph = False
            else:
                # GNN path
                X_t = torch.tensor(X_test, dtype=torch.float32)
                y_t = torch.tensor(y_test, dtype=torch.long)
                n = X_t.shape[0]
                edge_index = torch.tensor([[i,j] for i in range(n) for j in range(n) if i!=j],
                                          dtype=torch.long).t().contiguous()
                test_graph = Data(x=X_t, edge_index=edge_index, y=y_t)
                loader_test = GeoDataLoader([test_graph], batch_size=1, shuffle=False)
                is_graph = True

            ckpt_name = f"{model_type}_repFold{fold_count}.pt"
            single_model = build_single_model(model_type)
            load_checkpoint(single_model, ckpt_name)

            loss, acc, auc, sens, spec = evaluate_single_model(single_model, loader_test, is_graph=is_graph)

        elif model_type == "Ensemble":
            # Build a graph-based loader => the GNN in the ensemble needs data edge_index
            X_t = torch.tensor(X_test, dtype=torch.float32)
            y_t = torch.tensor(y_test, dtype=torch.long)
            n = X_t.shape[0]
            edge_index = torch.tensor([[i,j] for i in range(n) for j in range(n) if i!=j],
                                      dtype=torch.long).t().contiguous()
            test_graph = Data(x=X_t, edge_index=edge_index, y=y_t)
            loader_test = GeoDataLoader([test_graph], batch_size=1, shuffle=False)

            # Load all 3 model fold checkpoints
            models_dict = build_ensemble_for_fold(fold_count)
            # Evaluate ensemble
            loss, acc, auc, sens, spec = evaluate_ensemble(models_dict, loader_test)

        else:
            raise ValueError(f"Unsupported model: {model_type}")

        fold_losses.append(loss)
        fold_accuracies.append(acc)
        fold_aucs.append(auc)
        fold_sensitivities.append(sens)
        fold_specificities.append(spec)

        print(f"repFold{fold_count} => Loss={loss:.4f}, Acc={acc:.4f}, AUC={auc:.4f}, "
              f"Sens={sens:.4f}, Spec={spec:.4f}")
        fold_count += 1

    avg_loss = np.mean(fold_losses)
    avg_acc = np.mean(fold_accuracies)
    avg_auc = np.mean(fold_aucs)
    avg_sens = np.mean(fold_sensitivities)
    avg_spec = np.mean(fold_specificities)

    print(f"\n=== {model_type} repeated CV results: {n_splits}-fold, repeated {n_repeats} ===")
    print(f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, "
          f"Sens={avg_sens:.4f}, Spec={avg_spec:.4f}")
    
def evaluate_all(model_type, folds, repeats):
    # """
    # Expose evaluation as a function returning per-fold metrics and overall averages.
    # """
    # fold_metrics = []
    # overall = {}
    # Re‑use existing logic
    # fold_metrics, avg = main(model_type, n_splits=folds, n_repeats=repeats)
    # overall = {
    #     "loss": avg[0], "accuracy": avg[1], "auc": avg[2],
    #     "sensitivity": avg[3], "specificity": avg[4]
    # }
    return evaluate_main(model_type, folds, repeats)

if __name__ == "__main__":
    import argparse
    from models.gnn_model import PatientGNN
    from models.tabtransformer import TabTransformer
    from models.attention_mlp import AttentionMLP

    parser = argparse.ArgumentParser(description='Evaluate single or ensemble models with repeated CV')
    parser.add_argument('--model', type=str, required=True,
                        choices=["TabTransformer","AttentionMLP","GNN","Ensemble"],
                        help="Which model to evaluate or 'Ensemble' for a logit-averaged ensemble of TT, MLP, GNN.")
    parser.add_argument('--folds', type=int, default=2,
                        help="Number of folds (2 or 3, etc.)")
    parser.add_argument('--repeats', type=int, default=1,
                        help="Number of repeated runs (1, 2, 5, etc.)")
    args = parser.parse_args()

    main(args.model, n_splits=args.folds, n_repeats=args.repeats)
