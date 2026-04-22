import argparse
from pathlib import Path

import torch
from network_env import NetworkEnvironment
from traffic_generator import TrafficGenerator
from baseline_routing import BaselineRouting
from rl_agent import NetworkRoutingEnv, train_rl_agent
from gnn_model import GNNModel, GNNEmbedding, SUPPORTED_GNN_MODELS
from stable_baselines3 import PPO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
GNN_WEIGHTS = ARTIFACT_DIR / "gnn_pretrained.pt"
PPO_WEIGHTS = ARTIFACT_DIR / "ppo_gnn.zip"


def build_gnn_embedder(
    hidden_channels=64,
    embedding_dim=32,
    weights_path: Path = GNN_WEIGHTS,
    model_type="gcn",
):
    num_node_features = 3
    gnn = GNNModel(num_node_features, hidden_channels, embedding_dim, model_type=model_type)
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

def evaluate_methods(rl_model_path=PPO_WEIGHTS, rl_timesteps=0, gnn_model="gcn", gnn_weights=GNN_WEIGHTS):
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
    gnn_embedder = build_gnn_embedder(weights_path=Path(gnn_weights), model_type=gnn_model)
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

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate routing methods with optional GNN-enhanced RL.")
    parser.add_argument("--rl-model", type=str, default=str(PPO_WEIGHTS), help="Path to saved PPO policy.")
    parser.add_argument("--rl-timesteps", type=int, default=0, help="Train a fresh PPO policy if no saved model exists.")
    parser.add_argument("--gnn-model", choices=SUPPORTED_GNN_MODELS, default="gcn", help="GNN encoder architecture.")
    parser.add_argument("--gnn-weights", type=str, default=str(GNN_WEIGHTS), help="Path to pretrained GNN encoder weights.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_methods(
        rl_model_path=Path(args.rl_model),
        rl_timesteps=args.rl_timesteps,
        gnn_model=args.gnn_model,
        gnn_weights=Path(args.gnn_weights),
    )
