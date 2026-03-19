from pathlib import Path

import torch
from network_env import NetworkEnvironment
from traffic_generator import TrafficGenerator
from baseline_routing import BaselineRouting
from rl_agent import NetworkRoutingEnv, train_rl_agent
from gnn_model import GNNModel, GNNEmbedding
from stable_baselines3 import PPO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
GNN_WEIGHTS = ARTIFACT_DIR / "gnn_pretrained.pt"
PPO_WEIGHTS = ARTIFACT_DIR / "ppo_gnn.zip"


def build_gnn_embedder(hidden_channels=64, embedding_dim=32, weights_path: Path = GNN_WEIGHTS):
    num_node_features = 3
    gnn = GNNModel(num_node_features, hidden_channels, embedding_dim)
    if weights_path and weights_path.exists():
        state = torch.load(weights_path, map_location="cpu")
        try:
            gnn.load_state_dict(state)
            print(f"Loaded pretrained GNN weights from {weights_path}")
        except RuntimeError as exc:
            print(f"Could not load pretrained weights ({exc}); using random init instead.")
    else:
        print("No pretrained GNN weights found; using random initialization.")
    return GNNEmbedding(gnn)

def rollout_rl_policy(env, model):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
    reached_target = env.current_node == env.target
    latency = env.network_env.get_latency(env.path) if reached_target else np.inf
    return {
        'total_latency': latency,
        'successful_flows': 1 if reached_target else 0,
        'network_utilization': env.network_env.get_network_utilization(),
        'max_utilization': env.network_env.get_max_utilization(),
        'total_reward': total_reward,
        'path': env.path
    }

def evaluate_methods(rl_model_path=PPO_WEIGHTS, rl_timesteps=0):
    env = NetworkEnvironment()
    traffic_gen = TrafficGenerator(env)
    traffic_gen.set_fixed_flows()
    flows = traffic_gen.get_flows()

    baseline = BaselineRouting(env)

    # Shortest Path
    sp_results = baseline.simulate_traffic(flows)
    print("Shortest Path:", sp_results)

    # Random
    random_results = baseline.simulate_random_traffic(flows)
    print("Random:", random_results)

    # RL with GNN embeddings
    source, target = (0,0), (3,3)
    gnn_embedder = build_gnn_embedder()
    rl_env = NetworkRoutingEnv(env, source, target, gnn_embedder=gnn_embedder)

    rl_model = None
    if rl_model_path and Path(rl_model_path).exists():
        rl_model = PPO.load(rl_model_path, env=rl_env)
        print(f"Loaded PPO policy from {rl_model_path}")
    elif rl_timesteps > 0:
        rl_model = train_rl_agent(rl_env, total_timesteps=rl_timesteps)
        print("Trained fresh PPO policy for evaluation.")

    if rl_model is not None:
        rl_results = rollout_rl_policy(rl_env, rl_model)
    else:
        rl_results = {
            'total_latency': None,
            'successful_flows': 0,
            'network_utilization': None,
            'max_utilization': None,
            'total_reward': None,
            'path': []
        }
    print("RL (GNN-enhanced):", rl_results)

    # Plot (skip RL metrics if unavailable)
    methods = ['Shortest Path', 'Random']
    latencies = [sp_results['total_latency'], random_results['total_latency']]
    max_utils = [sp_results['max_utilization'], random_results['max_utilization']]
    if rl_results['total_latency'] is not None:
        methods.append('GNN + RL')
        latencies.append(rl_results['total_latency'])
        max_utils.append(rl_results['max_utilization'])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].bar(methods, latencies)
    ax[0].set_title('Total Latency')
    ax[1].bar(methods, max_utils)
    ax[1].set_title('Max Utilization')
    plt.tight_layout()
    chart_path = ARTIFACT_DIR / "evaluation_metrics.png"
    fig.savefig(chart_path)
    print(f"Saved evaluation plot to {chart_path}")

if __name__ == "__main__":
    evaluate_methods()
