# api/api_server.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import torch
import logging
from pathlib import Path
from io import BytesIO

from config.config import Config
from preprocessing.preprocessing import DataProcessor
from models.tabtransformer import TabTransformer
from models.attention_mlp import AttentionMLP
from models.gnn_model import PatientGNN
from torch_geometric.data import Data

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

app = FastAPI(title="DiaTrend API")

def build_single_model(model_type):
    if model_type == "TabTransformer":
        return TabTransformer(
            num_features=Config.NUM_FEATURES,
            num_classes=Config.NUM_CLASSES,
            dim=Config.TRANSFORMER_DIM,
            depth=Config.TRANSFORMER_DEPTH,
            heads=Config.TRANSFORMER_HEADS
        )
    elif model_type == "AttentionMLP":
        return AttentionMLP(
            input_dim=Config.NUM_FEATURES,
            hidden_dim=256,
            num_classes=Config.NUM_CLASSES
        )
    elif model_type == "GNN":
        return PatientGNN(num_features=Config.NUM_FEATURES, hidden_dim=128)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def load_fold_checkpoint(model_type, fold_index):
    """
    Loads e.g. 'TabTransformer_repFold{fold_index}.pt' from Config.CHECKPOINTS
    """
    ckpt_name = f"{model_type}_repFold{fold_index}.pt"
    ckpt_path = Config.CHECKPOINTS / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model_obj = build_single_model(model_type)
    state = torch.load(ckpt_path, map_location=Config.DEVICE)
    model_obj.load_state_dict(state)
    model_obj.to(Config.DEVICE)
    model_obj.eval()
    return model_obj

def single_inference(model_obj, features, model_type):
    """
    Single-sample inference using one model for the given features.
    """
    x_tensor = torch.tensor([features], dtype=torch.float32).to(Config.DEVICE)
    if model_type == "GNN":
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        data = Data(x=x_tensor, edge_index=edge_index).to(Config.DEVICE)
        with torch.no_grad():
            logits = model_obj(data)  # shape [1,2]
        pred_class = int(logits.argmax(dim=1).item())
    else:
        with torch.no_grad():
            logits = model_obj(x_tensor)  # shape [1,2]
        pred_class = int(logits.argmax(dim=1).item())
    return pred_class

def ensemble_inference(features, fold_index):
    """
    Load TabTransformer, AttentionMLP, GNN for this fold_index,
    average logits for single-sample inference.
    """
    import torch
    from torch_geometric.data import Data

    x_tensor = torch.tensor([features], dtype=torch.float32).to(Config.DEVICE)
    edge_index = torch.tensor([[0],[0]], dtype=torch.long).to(Config.DEVICE)
    data = Data(x=x_tensor, edge_index=edge_index).to(Config.DEVICE)

    # Load each model
    model_tt = load_fold_checkpoint("TabTransformer", fold_index)
    model_mlp = load_fold_checkpoint("AttentionMLP", fold_index)
    model_gnn = load_fold_checkpoint("GNN", fold_index)

    with torch.no_grad():
        out_tt = model_tt(x_tensor)
        out_mlp = model_mlp(x_tensor)
        out_gnn = model_gnn(data)

        avg_logits = (out_tt + out_mlp + out_gnn) / 3.0
        pred_class = int(avg_logits.argmax(dim=1).item())
    return pred_class

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("Ensemble", enum=["TabTransformer","AttentionMLP","GNN","Ensemble"]),
    fold_index: int = Query(1, gt=0, description="Repeated fold index (>=1)"),
):
    """
    Example usage:
      curl -X POST "http://127.0.0.1:8000/predict?model=Ensemble&fold_index=2" \
           -F "file=@Subject15.xlsx"
    """
    try:
        # read xlsx contents
        contents = await file.read()
        file_obj = BytesIO(contents)
        file_obj.name = file.filename if file.filename else "uploaded.xlsx"

        # parse features
        processor = DataProcessor()
        demo_df = processor._load_demographics()
        features, _ = processor.process_file(file_obj, demo_df, filename=file_obj.name)

        # pick model path
        if model in ["TabTransformer","AttentionMLP","GNN"]:
            model_obj = load_fold_checkpoint(model, fold_index)
            pred_class = single_inference(model_obj, features, model)
        else:
            # 'Ensemble'
            pred_class = ensemble_inference(features, fold_index)

        return {"prediction": pred_class}

    except FileNotFoundError as fnf:
        logger.error(str(fnf))
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
