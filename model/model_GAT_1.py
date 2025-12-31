import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATFeatureExtractor(nn.Module):
    def __init__(self, observation_space, hidden_size=64):
        super(GATFeatureExtractor, self).__init__()
        self.observation_space = observation_space
        self.hidden_size = hidden_size

        # We assume that the observation space is a Box space with shape (num_nodes, num_features)
        num_nodes, num_features = observation_space.shape

        # Define the GAT layers
        self.gat1 = GATConv(num_features, hidden_size, heads=8, dropout=0.6)
        self.gat2 = GATConv(hidden_size * 8, hidden_size, dropout=0.6)

        # Compute the output size of the GAT layers
        x = torch.zeros((1, num_nodes, num_features))
        x = self.gat1(x)
        x = self.gat2(x)
        self.output_size = x.size(-1)

    def forward(self, x):
        # Convert the observation x to a PyTorch Geometric Data object
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x = x.transpose(1, 2)

        # Apply the GAT layers
        x = F.elu(self.gat1(x))
        x = F.elu(self.gat2(x))

        # Reshape the output tensor to (batch_size, output_size)
        x = x.view(x.size(0), -1)

        return x
