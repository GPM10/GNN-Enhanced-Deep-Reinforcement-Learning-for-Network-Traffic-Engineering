import random
from network_env import NetworkEnvironment

class TrafficGenerator:
    def __init__(self, env: NetworkEnvironment):
        self.env = env
        self.flows = []

    def generate_random_flows(self, num_flows=5):
        """
        Generate random traffic flows.
        """
        nodes = list(self.env.G.nodes())
        self.flows = []
        for _ in range(num_flows):
            source = random.choice(nodes)
            target = random.choice(nodes)
            while target == source:
                target = random.choice(nodes)
            amount = random.randint(1, 5)
            self.flows.append((source, target, amount))

    def set_fixed_flows(self):
        """
        Set fixed flows as per example.
        Assuming 4x4 grid, nodes (0,0) to (3,3).
        Map: Node 1 -> (0,0), Node 10 -> (2,1), etc.? Wait, better to use indices.
        Perhaps renumber or use tuples.
        For simplicity, use tuples.
        Node 1: (0,0), Node 10: (2,1)? Let's count: 0 to 15.
        (0,0)=0, (0,1)=1, ..., (3,3)=15.
        So Node 1 -> (0,0), Node 10 -> (2,1), Node 3 -> (0,2), Node 12 -> (3,0), Node 6 -> (1,2), Node 15 -> (3,3)
        """
        self.flows = [
            ((0,0), (2,1), 2),  # Node 1 -> Node 10
            ((0,2), (3,0), 3),  # Node 3 -> Node 12
            ((1,2), (3,3), 1)   # Node 6 -> Node 15
        ]

    def get_flows(self):
        """
        Return the list of flows.
        """
        return self.flows