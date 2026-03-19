import random
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from network_env import NetworkEnvironment
from traffic_generator import TrafficGenerator
from baseline_routing import BaselineRouting
from gnn_model import networkx_to_pyg_data


def _node_utilization(env: NetworkEnvironment, node) -> float:
    """Average utilization of edges incident to node."""
    loads = []
    for _, _, data in env.G.edges(node, data=True):
        capacity = data.get("capacity", 0)
        load = data.get("load", 0)
        util = (load / capacity) if capacity else 0.0
        loads.append(util)
    return float(np.mean(loads)) if loads else 0.0


def generate_congestion_samples(
    num_samples: int = 200,
    flows_per_sample: int = 6,
    random_flows: bool = True,
    seed: Optional[int] = None,
) -> List[Data]:
    """
    Run the classical simulator multiple times and capture node-level congestion labels.
    Returns a list of PyG Data objects with `x`, `edge_index`, and node targets `y`.
    """
    rng = random.Random(seed)
    samples: List[Data] = []

    for _ in range(num_samples):
        env = NetworkEnvironment()
        traffic_gen = TrafficGenerator(env)
        if random_flows:
            traffic_gen.generate_random_flows(num_flows=flows_per_sample)
        else:
            traffic_gen.set_fixed_flows()
        flows = traffic_gen.get_flows()

        baseline = BaselineRouting(env)
        baseline.simulate_traffic(flows)

        data, node_to_idx = networkx_to_pyg_data(env.G, return_mapping=True)
        labels = torch.zeros(len(node_to_idx), dtype=torch.float)
        for node, idx in node_to_idx.items():
            labels[idx] = _node_utilization(env, node)
        data.y = labels
        samples.append(data)

    return samples

