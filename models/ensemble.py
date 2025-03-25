# models/ensemble.py
import torch
from config.config import Config
from models.tabtransformer import TabTransformer
from models.attention_mlp import AttentionMLP
from models.gnn_model import PatientGNN
from torch_geometric.data import Data

def load_tabtransformer():
    model = TabTransformer(
        num_features=Config.NUM_FEATURES,
        num_classes=Config.NUM_CLASSES,
        dim=Config.TRANSFORMER_DIM,
        depth=Config.TRANSFORMER_DEPTH,
        heads=Config.TRANSFORMER_HEADS,
    )
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATHS["TabTransformer"], map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

def load_attention_mlp():
    model = AttentionMLP(
        input_dim=Config.NUM_FEATURES,
        hidden_dim=256,
        num_classes=Config.NUM_CLASSES
    )
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATHS["AttentionMLP"], map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

def load_gnn():
    model = PatientGNN(
        num_features=Config.NUM_FEATURES,
        hidden_dim=128
    )
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATHS["GNN"], map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

def ensemble_predict(features):
    tensor_input = torch.tensor([features], dtype=torch.float32).to(Config.DEVICE)
    x = torch.tensor([features], dtype=torch.float32)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index).to(Config.DEVICE)
    
    model_tt = load_tabtransformer()
    model_attn = load_attention_mlp()
    model_gnn = load_gnn()
    
    with torch.no_grad():
        out_tt = model_tt(tensor_input)
        out_attn = model_attn(tensor_input)
        out_gnn = model_gnn(graph)
        out_gnn = out_gnn[0].unsqueeze(0)
    avg_out = (out_tt + out_attn + out_gnn) / 3
    pred = int(avg_out.argmax(dim=1).item())
    return pred
