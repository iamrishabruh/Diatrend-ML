# File: models/gnn_model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PatientGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        """
        GNN for classification. Typically requires a graph structure (edge_index).
        """
        super(PatientGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 2)

    def forward(self, data):
        # data.x: [num_nodes, num_features]
        # data.edge_index: [2, num_edges]
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return self.classifier(x)
