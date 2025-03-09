# File: models/attention_mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        """
        MLP with a simple attention mechanism on the hidden representation.
        """
        super(AttentionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: [batch_size, input_dim]
        x = F.relu(self.fc1(x))    # [batch_size, hidden_dim]
        # Compute attention weights for each feature dimension
        attn_weights = F.softmax(self.attention(x), dim=1)  # [batch_size, hidden_dim, 1]
        weighted = x * attn_weights  # element-wise
        out = self.fc2(weighted.sum(dim=1))  # sum across feature dimension
        return out
