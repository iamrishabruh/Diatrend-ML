# models/ensemble.py

import torch
from config.config import Config
from models.tabtransformer import TabTransformer
from models.attention_mlp import AttentionMLP
from models.gnn_model import PatientGNN
from torch_geometric.data import Data

###############################################################################
# LOADING INDIVIDUAL MODELS
###############################################################################
def load_tabtransformer():
    model = TabTransformer(
        num_features=Config.NUM_FEATURES,
        num_classes=Config.NUM_CLASSES,
        dim=Config.TRANSFORMER_DIM,
        depth=Config.TRANSFORMER_DEPTH,
        heads=Config.TRANSFORMER_HEADS
    )
    state = torch.load(Config.CHECKPOINT_PATHS["TabTransformer"], map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.to(Config.DEVICE)
    model.eval()
    return model

def load_attention_mlp():
    model = AttentionMLP(
        input_dim=Config.NUM_FEATURES,
        hidden_dim=64,
        num_classes=Config.NUM_CLASSES
    )
    state = torch.load(Config.CHECKPOINT_PATHS["AttentionMLP"], map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.to(Config.DEVICE)
    model.eval()
    return model

def load_gnn():
    model = PatientGNN(
        num_features=Config.NUM_FEATURES,
        hidden_dim=128
    )
    state = torch.load(Config.CHECKPOINT_PATHS["GNN"], map_location=Config.DEVICE)
    model.load_state_dict(state, strict=False)  # in case of minor architecture changes
    model.to(Config.DEVICE)
    model.eval()
    return model

###############################################################################
# SINGLE-SAMPLE ENSEMBLE (returns discrete label)
###############################################################################
def ensemble_predict(features):
    """
    Predict a single sample. Creates a trivial single-node graph for GNN 
    and direct inference for TabTransformer & AttentionMLP, then averages logits.
    Returns discrete class (0 or 1).
    """
    tensor_input = torch.tensor([features], dtype=torch.float32).to(Config.DEVICE)

    # GNN single-node graph
    x = torch.tensor([features], dtype=torch.float32)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index).to(Config.DEVICE)
    
    # Load each model
    model_tt = load_tabtransformer()
    model_attn = load_attention_mlp()
    model_gnn = load_gnn()
    
    with torch.no_grad():
        logits_tt = model_tt(tensor_input)
        logits_attn = model_attn(tensor_input)
        logits_gnn = model_gnn(graph)[0].unsqueeze(0)  # single node

        avg_logits = (logits_tt + logits_attn + logits_gnn) / 3.0
        pred_class = int(avg_logits.argmax(dim=1).item())

    return pred_class

###############################################################################
# BATCH ENSEMBLE (returns probabilities)
###############################################################################
def ensemble_predict_batch(X):
    """
    Accepts a 2D numpy array or list of shape [N, NUM_FEATURES].
    Returns class probabilities for each sample using:
      - fully connected graph for GNN
      - normal forward for TabTransformer & AttentionMLP
      - softmax over averaged logits
    """
    tensor_input = torch.tensor(X, dtype=torch.float32).to(Config.DEVICE)
    N = tensor_input.shape[0]

    # Build a fully connected graph for GNN
    edges = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edges.append([i, j])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    graph = Data(x=tensor_input, edge_index=edge_index).to(Config.DEVICE)

    # Load each model
    model_tt = load_tabtransformer()
    model_attn = load_attention_mlp()
    model_gnn = load_gnn()

    with torch.no_grad():
        logits_tt = model_tt(tensor_input)      # [N, 2]
        logits_attn = model_attn(tensor_input)  # [N, 2]
        logits_gnn = model_gnn(graph)           # [N, 2]

        # Average the 3 sets of logits
        avg_logits = (logits_tt + logits_attn + logits_gnn) / 3.0

        # Return probabilities via softmax
        probs = torch.softmax(avg_logits, dim=1)  # shape [N, 2]
    return probs.cpu().numpy()
