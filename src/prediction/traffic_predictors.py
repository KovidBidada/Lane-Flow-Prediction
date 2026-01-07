"""
Traffic Prediction Module
--------------------------
Implements an advanced LSTM-based model for short-term traffic flow prediction.
Features:
- Stacked LSTM layers
- Bidirectional support
- Layer Normalization
- Attention mechanism (optional)
- Multi-step prediction (via prediction_horizon)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class TrafficPredictor(nn.Module):
    """
    Advanced LSTM for traffic prediction:
    - Stacked LSTM layers
    - Bidirectional
    - LayerNorm
    - Optional Attention mechanism
    - Sequence-to-sequence (multi-step) prediction
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_size: int = 1,
        sequence_length: int = 12,
        prediction_horizon: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super(TrafficPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        # Define LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # Attention layer
        if use_attention:
            self.attn = nn.Linear(hidden_size * self.num_directions, 1)

        # Fully connected output: supports multi-step output
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size * prediction_horizon)

        logger.info(
            f"TrafficPredictor initialized | sequence_length={sequence_length}, "
            f"prediction_horizon={prediction_horizon}, bidirectional={bidirectional}, "
            f"use_attention={use_attention}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Tensor of shape [batch_size, seq_len, input_size]

        Returns:
            Tensor of shape [batch_size, prediction_horizon, output_size]
        """
        batch_size = x.size(0)

        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # out: [batch, seq_len, hidden*directions]
        out = self.layer_norm(out)

        if self.use_attention:
            # Compute attention weights over time dimension
            attn_weights = F.softmax(self.attn(out), dim=1)  # [batch, seq_len, 1]
            out = (out * attn_weights).sum(dim=1)            # [batch, hidden*directions]
        else:
            # Use last timestep output if no attention
            out = out[:, -1, :]

        # Predict multi-step (horizon) output
        out = self.fc(out)  # [batch, output_size * horizon]
        out = out.view(batch_size, self.prediction_horizon, -1)  # reshape

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GCNTrafficPredictor(nn.Module):
    """
    Advanced GCN for traffic prediction:
    - Multiple GCN layers
    - Optional GAT attention layers
    - Skip connections
    - Global pooling for graph-level output
    """
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, output_dim=1,
                 num_layers=3, use_attention=False, dropout=0.2):
        super(GCNTrafficPredictor, self).__init__()
        
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.dropout = dropout
        
        # Define GCN / GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            if use_attention:
                self.convs.append(GATConv(in_dim, out_dim, heads=1, concat=True, dropout=dropout))
            else:
                self.convs.append(GCNConv(in_dim, out_dim))
        
        # Optional layer norm for each layer
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers-1)])
        
    def forward(self, x, edge_index, batch=None):
        """
        x: [num_nodes_total, input_dim] Node features
        edge_index: [2, num_edges] Graph connectivity
        batch: [num_nodes_total] Batch assignment if using multiple graphs
        """
        for i, conv in enumerate(self.convs):
            x_res = x  # Skip connection
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.norms[i](x)
                x = x + x_res  # Skip connection
        
        # If batch provided, do graph-level pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x
