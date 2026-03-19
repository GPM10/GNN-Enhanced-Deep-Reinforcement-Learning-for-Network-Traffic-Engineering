import argparse
from pathlib import Path

import torch

from network_env import NetworkEnvironment
from rl_agent import NetworkRoutingEnv, train_rl_agent
from traffic_generator import TrafficGenerator
from baseline_routing import BaselineRouting
from gnn_model import GNNModel, GNNEmbedding


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
GNN_WEIGHTS = ARTIFACT_DIR / "gnn_pretrained.pt"
PPO_WEIGHTS = ARTIFACT_DIR / "ppo_gnn.zip"


def build_gnn_embedder(hidden_channels=64, embedding_dim=32, weights_path: Path = GNN_WEIGHTS):
    """
    Create a lightweight GNN embedder for the routing environment.
    """
    num_node_features = 3  # [x_norm, y_norm, avg_util]
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent with GNN-enhanced observations.")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total PPO training timesteps.")
    parser.add_argument("--save-model", type=str, default=str(PPO_WEIGHTS), help="Path to save trained PPO policy.")
    parser.add_argument("--source", type=int, nargs=2, default=(0, 0), help="Source node coordinates.")
    parser.add_argument("--target", type=int, nargs=2, default=(3, 3), help="Target node coordinates.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Create network
    env = NetworkEnvironment()
    traffic_gen = TrafficGenerator(env)
    traffic_gen.set_fixed_flows()
    flows = traffic_gen.get_flows()

    # Baseline
    baseline = BaselineRouting(env)
    baseline_results = baseline.simulate_traffic(flows)
    print("Baseline Results:", baseline_results)

    # RL Training with GNN-enhanced observations
    source = tuple(args.source)
    target = tuple(args.target)
    gnn_embedder = build_gnn_embedder()
    rl_env = NetworkRoutingEnv(env, source, target, gnn_embedder=gnn_embedder)
    model = train_rl_agent(rl_env, total_timesteps=args.timesteps)

    # Save PPO policy
    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f"Saved PPO policy to {save_path}")

    # Test RL
    obs, _ = rl_env.reset()
    done = False
    step_count = 0
    max_steps = 200
    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = rl_env.step(action)
        step_count += 1
    print("RL Path:", rl_env.path)

if __name__ == "__main__":
    main()
