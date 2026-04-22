# GNN-Enhanced Deep Reinforcement Learning for Network Traffic Engineering

This project demonstrates the use of Graph Neural Networks (GNNs) combined with Deep Reinforcement Learning (RL) to optimize routing in network traffic engineering.

## Goal
Learn routing policies in a network graph that reduce congestion and latency.

## Stages
1. **Network Environment**: Build a simulator using NetworkX.
2. **Baseline Routing**: Implement shortest path routing.
3. **Reinforcement Learning**: Add RL agent using stable-baselines3.
4. **Graph Neural Network**: Integrate GNN for topology-aware representations.
5. **Evaluation**: Compare methods and visualize results.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
- Run `train.py` to train the RL agent.
- Run `evaluate.py` to compare routing methods.
- Run `visualize.py` to plot the network.
- Run `gnn_pretrain.py` to generate congestion-labelled graphs and pretrain the GNN encoder. The resulting weights are stored in `gnn_rl_network/artifacts/gnn_pretrained.pt` and are automatically loaded by `train.py`/`evaluate.py` when present.
- Select the encoder with `--gnn-model gcn`, `--gnn-model gat`, or `--gnn-model graphsage`. GCN remains the default.

Examples:
```bash
python gnn_pretrain.py --gnn-model gat --output gnn_gat_pretrained.pt
python train.py --gnn-model gat --gnn-weights artifacts/gnn_gat_pretrained.pt
python evaluate.py --gnn-model graphsage
```

## Files
- `network_env.py`: Network simulator.
- `traffic_generator.py`: Generate traffic flows.
- `baseline_routing.py`: Shortest path and random routing.
- `rl_agent.py`: RL environment and training.
- `gnn_model.py`: GNN model for embeddings.
- `train.py`: Training script.
- `evaluate.py`: Evaluation and comparison.
- `visualize.py`: Visualization tools.
