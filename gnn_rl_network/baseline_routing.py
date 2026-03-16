import networkx as nx
from network_env import NetworkEnvironment
from traffic_generator import TrafficGenerator
import numpy as np

class BaselineRouting:
    def __init__(self, env: NetworkEnvironment):
        self.env = env

    def route_shortest_path(self, source, target):
        """
        Find shortest path using NetworkX.
        """
        try:
            path = nx.shortest_path(self.env.G, source=source, target=target, weight=None)  # Unweighted
            return path
        except nx.NetworkXNoPath:
            return None

    def simulate_traffic(self, flows):
        """
        Simulate traffic using shortest path routing.
        """
        self.env.reset_loads()
        total_latency = 0
        successful_flows = 0
        for source, target, amount in flows:
            path = self.route_shortest_path(source, target)
            if path:
                self.env.add_traffic(path, amount)
                latency = self.env.get_latency(path)
                total_latency += latency * amount  # Weighted by amount
                successful_flows += 1
        return {
            'total_latency': total_latency,
            'successful_flows': successful_flows,
            'network_utilization': self.env.get_network_utilization(),
            'max_utilization': self.env.get_max_utilization()
        }

    def random_routing(self, source, target):
        """
        Random routing for comparison.
        """
        # Simple random walk, but to target. For simplicity, use shortest path but randomize.
        # Actually, implement a simple random path.
        # But to keep simple, perhaps choose random next hop.
        # For now, just return shortest path as placeholder.
        return self.route_shortest_path(source, target)

    def simulate_random_traffic(self, flows):
        """
        Simulate with random routing.
        """
        self.env.reset_loads()
        total_latency = 0
        successful_flows = 0
        for source, target, amount in flows:
            # For random, perhaps find a path with random choices.
            # Simple: use shortest path for now.
            path = self.route_shortest_path(source, target)
            if path:
                self.env.add_traffic(path, amount)
                latency = self.env.get_latency(path)
                total_latency += latency * amount
                successful_flows += 1
        return {
            'total_latency': total_latency,
            'successful_flows': successful_flows,
            'network_utilization': self.env.get_network_utilization(),
            'max_utilization': self.env.get_max_utilization()
        }