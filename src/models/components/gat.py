import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_heads: int = 1,
            dropout: float = 0.5,
    ) -> None:
        super(GAT, self).__init__()

        self.conv1 = GATConv(input_size, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_size * num_heads, output_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
