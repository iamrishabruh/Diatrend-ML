# models/gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

class PatientGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=128, dropout=0.2):
        """
        Graph Neural Network model for capturing patient similarity with increased depth.
        Uses three GCNConv layers with batch normalization and dropout.
        """
        super(PatientGNN, self).__init__()
        logger.debug(f"Initializing PatientGNN with num_features={num_features}, hidden_dim={hidden_dim}")
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.classifier = torch.nn.Linear(hidden_dim, 2)

    def forward(self, data):
        logger.debug("Forward pass of PatientGNN.")
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(F.relu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(F.relu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(F.relu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.classifier(x)
        logger.debug(f"PatientGNN output shape: {out.shape}")
        return out
