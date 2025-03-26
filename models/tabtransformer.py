# models/tabtransformer.py
import torch
import torch.nn as nn
from config.config import Config

class TabTransformer(nn.Module):
    def __init__(self, 
                 num_features=Config.NUM_FEATURES, 
                 num_classes=Config.NUM_CLASSES, 
                 dim=Config.TRANSFORMER_DIM, 
                 depth=Config.TRANSFORMER_DEPTH, 
                 heads=Config.TRANSFORMER_HEADS, 
                 dropout=0.3):
        """
        TabTransformer for tabular data with increased capacity.
        Uses hyperparameters defined in Config.
        """
        super(TabTransformer, self).__init__()
        self.embed = nn.Linear(num_features, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, num_features]
        x = x.unsqueeze(1)          # [batch_size, 1, num_features]
        x = self.embed(x)           # [batch_size, 1, dim]
        x = self.transformer(x)     # [batch_size, 1, dim]
        x = x.mean(dim=1)          # [batch_size, dim]
        return self.classifier(x)
