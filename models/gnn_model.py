import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import logging
from config.config import Config

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

class PatientGNN(torch.nn.Module):
    def __init__(self, 
                 num_features, 
                 hidden_dim=Config.GNN_HIDDEN_DIM, 
                 dropout=Config.GNN_DROPOUT, 
                 num_layers=Config.GNN_NUM_LAYERS):
        """
        Graph Neural Network model for capturing patient similarity.
        Dynamically builds 'num_layers' of GCNConv with BatchNorm and dropout.
        """
        super(PatientGNN, self).__init__()
        logger.debug(f"Initializing PatientGNN with num_features={num_features}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        # First layer: input to hidden_dim
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        # Remaining layers: hidden_dim to hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.classifier = torch.nn.Linear(hidden_dim, Config.NUM_CLASSES)

    def forward(self, data):
        # logger.debug("Forward pass of PatientGNN.")
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(F.relu(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.classifier(x)
        # logger.debug(f"PatientGNN output shape: {out.shape}")
        return out
