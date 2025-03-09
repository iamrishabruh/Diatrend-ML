# File: models/tabtransformer.py

import torch
import torch.nn as nn

class TabTransformer(nn.Module):
    def __init__(self, num_features, num_classes, dim=32, depth=4):
        """
        TabTransformer for tabular data.
        """
        super(TabTransformer, self).__init__()
        self.embed = nn.Linear(num_features, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, num_features], but we treat it as [batch_size, seq_len=1, dim]
        x = x.unsqueeze(1)  # shape: [batch_size, 1, num_features]
        x = self.embed(x)   # shape: [batch_size, 1, dim]
        x = self.transformer(x)  # shape: [batch_size, 1, dim]
        x = x.mean(dim=1)   # [batch_size, dim]
        return self.classifier(x)
