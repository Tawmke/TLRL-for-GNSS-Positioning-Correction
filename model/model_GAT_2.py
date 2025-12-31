import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GraphAttentionNetwork(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphAttentionNetwork, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GATConv(16, num_classes, heads=8)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class TemporalCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 10, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class PPOFeatureExtractor(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PPOFeatureExtractor, self).__init__()
        self.graph_network = GraphAttentionNetwork(observation_space["graph"].num_features, 64)
        self.temporal_network = TemporalCNN(observation_space["temporal"].shape[0], 64)

    def forward(self, observation):
        x1 = self.graph_network(observation["graph"].x, observation["graph"].edge_index)
        x2 = self.temporal_network(observation["temporal"].unsqueeze(0))
        x = torch.cat((x1, x2), dim=1)
        return x
