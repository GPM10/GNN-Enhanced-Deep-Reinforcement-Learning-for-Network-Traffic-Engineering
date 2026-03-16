import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np

class GNNModel(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

def networkx_to_pyg_data(G, node_features=None):
    """
    Convert NetworkX graph to PyG Data.
    node_features: dict of node to feature vector
    """
    if node_features is None:
        # Default: position
        node_features = {node: np.array([node[0], node[1]]) for node in G.nodes()}

    x = torch.tensor([node_features[node] for node in G.nodes()], dtype=torch.float)
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    # Add reverse edges if undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Edge features: capacity, load
    edge_attr = []
    for u, v in G.edges():
        capacity = G[u][v]['capacity']
        load = G[u][v]['load']
        edge_attr.append([capacity, load])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    # Duplicate for reverse
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# For RL, perhaps embed the graph and use for state.

class GNNEmbedding:
    def __init__(self, model):
        self.model = model

    def get_embeddings(self, G):
        data = networkx_to_pyg_data(G)
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index)
        return embeddings  # Node embeddings