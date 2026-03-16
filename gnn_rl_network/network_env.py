import networkx as nx
import random
import numpy as np

class NetworkEnvironment:
    def __init__(self, grid_size=4):
        """
        Initialize a grid network with capacities and loads.
        """
        self.G = nx.grid_2d_graph(grid_size, grid_size)
        self._initialize_links()

    def _initialize_links(self):
        """
        Set random capacities and initial loads for each link.
        """
        for u, v in self.G.edges():
            self.G[u][v]['capacity'] = random.randint(10, 20)
            self.G[u][v]['load'] = 0

    def get_graph(self):
        """
        Return the network graph.
        """
        return self.G

    def reset_loads(self):
        """
        Reset all link loads to 0.
        """
        for u, v in self.G.edges():
            self.G[u][v]['load'] = 0

    def add_traffic(self, path, amount=1):
        """
        Add traffic along a path, updating link loads.
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.G.has_edge(u, v):
                self.G[u][v]['load'] += amount
            else:
                # Assuming undirected, but grid is undirected
                self.G[v][u]['load'] += amount

    def get_link_utilization(self, u, v):
        """
        Get utilization of a link (load / capacity).
        """
        if self.G.has_edge(u, v):
            load = self.G[u][v]['load']
            capacity = self.G[u][v]['capacity']
            return load / capacity if capacity > 0 else 0
        return 0

    def get_network_utilization(self):
        """
        Get average utilization across all links.
        """
        utilizations = [self.get_link_utilization(u, v) for u, v in self.G.edges()]
        return np.mean(utilizations) if utilizations else 0

    def get_max_utilization(self):
        """
        Get maximum link utilization.
        """
        utilizations = [self.get_link_utilization(u, v) for u, v in self.G.edges()]
        return max(utilizations) if utilizations else 0

    def get_latency(self, path):
        """
        Estimate latency for a path based on link loads.
        Simple model: latency = sum(1 + load/capacity) for each link.
        """
        latency = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            util = self.get_link_utilization(u, v)
            latency += 1 + util  # Base latency + congestion penalty
        return latency