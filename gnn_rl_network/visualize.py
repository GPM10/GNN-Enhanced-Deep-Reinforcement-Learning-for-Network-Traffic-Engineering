import matplotlib.pyplot as plt
import networkx as nx
from network_env import NetworkEnvironment

def visualize_network(G, title="Network Topology"):
    pos = {node: node for node in G.nodes()}  # Use node positions
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    labels = {(u,v): f"{G[u][v]['load']}/{G[u][v]['capacity']}" for u,v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, labels)
    plt.title(title)
    plt.show()

def plot_metrics(methods, latencies, utils):
    fig, ax = plt.subplots(1, 2)
    ax[0].bar(methods, latencies)
    ax[0].set_title('Latency')
    ax[1].bar(methods, utils)
    ax[1].set_title('Utilization')
    plt.show()

# Example usage
if __name__ == "__main__":
    env = NetworkEnvironment()
    visualize_network(env.G)