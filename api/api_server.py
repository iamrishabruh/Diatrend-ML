from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import logging
from pathlib import Path
from io import BytesIO

from config.config import Config
from preprocessing.preprocessing import DataProcessor
from models.tabtransformer import TabTransformer  # Added import

app = FastAPI()
logger = logging.getLogger("api")

class PredictionModel:
    def __init__(self):
        self.model = None
        self.processor = DataProcessor()
        
    def load_model(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError("Model checkpoint missing")
        self.model = TabTransformer(
            num_features=Config.NUM_FEATURES,
            num_classes=Config.NUM_CLASSES,
            dim=Config.TRANSFORMER_DIM,
            depth=Config.TRANSFORMER_DEPTH,
            heads=Config.TRANSFORMER_HEADS
        )
        self.model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        self.model.to(Config.DEVICE)
        self.model.eval()
        
model_wrapper = PredictionModel()
model_wrapper.load_model(Config.CHECKPOINTS / "TabTransformer_best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_obj = BytesIO(contents)
        if hasattr(file, "filename"):
            file_obj.name = file.filename
        features = model_wrapper.processor.process_file(file_obj, 
                            model_wrapper.processor._load_demographics())
        tensor = torch.tensor([features], dtype=torch.float32).to(Config.DEVICE) ## TYPE ERROR PREDICTION
        with torch.no_grad():
            output = model_wrapper.model(tensor)
        return {"prediction": int(output.argmax().item())}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
