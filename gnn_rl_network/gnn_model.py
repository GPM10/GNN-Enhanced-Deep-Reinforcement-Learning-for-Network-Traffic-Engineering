import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from typing import Dict, Any, Tuple

SUPPORTED_GNN_MODELS = ("gcn", "gat", "graphsage")


class GNNModel(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, model_type="gcn", gat_heads=4):
        super().__init__()
        self.model_type = _normalize_model_type(model_type)
        self.embedding_dim = num_classes

        if self.model_type == "gcn":
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, num_classes)
        elif self.model_type == "gat":
            self.conv1 = GATConv(num_node_features, hidden_channels, heads=gat_heads, concat=False)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=gat_heads, concat=False)
            self.conv3 = GATConv(hidden_channels, num_classes, heads=1, concat=False)
        elif self.model_type == "graphsage":
            self.conv1 = SAGEConv(num_node_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x


def _normalize_model_type(model_type: str) -> str:
    normalized = model_type.lower().replace("_", "").replace("-", "")
    aliases = {
        "gcn": "gcn",
        "gat": "gat",
        "sage": "graphsage",
        "graphsage": "graphsage",
    }
    if normalized not in aliases:
        options = ", ".join(SUPPORTED_GNN_MODELS)
        raise ValueError(f"Unsupported GNN model '{model_type}'. Choose one of: {options}.")
    return aliases[normalized]


def _default_feature_vector(G: nx.Graph, node: Any, idx: int, total_nodes: int,
                            coord_stats: Tuple[int, int, bool]) -> np.ndarray:
    """Build a simple feature vector [x_norm, y_norm, avg_utilization]."""
    max_x, max_y, has_coords = coord_stats
    if has_coords and isinstance(node, tuple) and len(node) == 2:
        norm_x = node[0] / max(1, max_x)
        norm_y = node[1] / max(1, max_y)
    else:
        # Fall back to index-based encoding when coordinates are unavailable
        norm_x = idx / max(1, total_nodes - 1)
        norm_y = norm_x

    incident_edges = list(G.edges(node, data=True))
    if incident_edges:
        utils = []
        for _, _, data in incident_edges:
            capacity = data.get('capacity', 0)
            load = data.get('load', 0)
            util = (load / capacity) if capacity else 0.0
            utils.append(util)
        avg_util = float(np.mean(utils))
    else:
        avg_util = 0.0

    return np.array([norm_x, norm_y, avg_util], dtype=np.float32)


def _coordinate_stats(G: nx.Graph) -> Tuple[int, int, bool]:
    coord_nodes = [n for n in G.nodes() if isinstance(n, tuple) and len(n) == 2]
    if not coord_nodes:
        return 1, 1, False
    max_x = max(n[0] for n in coord_nodes)
    max_y = max(n[1] for n in coord_nodes)
    return max_x, max_y, True


def networkx_to_pyg_data(G, node_features=None, return_mapping=False):
    """
    Convert NetworkX graph to PyG Data.
    node_features: dict of node to feature vector
    return_mapping: optionally return node->index map for downstream lookup
    """
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    if node_features is None:
        coord_stats = _coordinate_stats(G)
        feature_lookup: Dict[Any, np.ndarray] = {}
        for idx, node in enumerate(nodes):
            feature_lookup[node] = _default_feature_vector(G, node, idx, len(nodes), coord_stats)
    else:
        feature_lookup = node_features

    x = torch.tensor([feature_lookup[node] for node in nodes], dtype=torch.float)

    directed_edges = []
    edge_attrs = []
    for u, v in G.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        directed_edges.append((u_idx, v_idx))
        directed_edges.append((v_idx, u_idx))

        capacity = G[u][v].get('capacity', 0)
        load = G[u][v].get('load', 0)
        edge_attrs.append([capacity, load])
        edge_attrs.append([capacity, load])

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if return_mapping:
        return data, node_to_idx
    return data

# For RL, perhaps embed the graph and use for state.

class GNNEmbedding:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = getattr(model, 'embedding_dim', None) or model.conv3.out_channels

    def get_embeddings(self, G) -> Dict[Any, np.ndarray]:
        data, node_to_idx = networkx_to_pyg_data(G, return_mapping=True)
        data = data.to(self.device)
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index)
        embeddings = embeddings.cpu().numpy()
        return {node: embeddings[idx] for node, idx in node_to_idx.items()}
