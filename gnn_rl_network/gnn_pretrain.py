import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader

from gnn_model import GNNModel, SUPPORTED_GNN_MODELS
from gnn_dataset import generate_congestion_samples


class GNNCongestionRegressor(nn.Module):
    def __init__(self, hidden_channels=64, embedding_dim=32, model_type="gcn"):
        super().__init__()
        self.encoder = GNNModel(
            num_node_features=3,
            hidden_channels=hidden_channels,
            num_classes=embedding_dim,
            model_type=model_type,
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, data):
        embeddings = self.encoder(data.x, data.edge_index)
        preds = self.head(embeddings).squeeze(-1)
        return preds


def train_gnn(args):
    samples = generate_congestion_samples(
        num_samples=args.samples,
        flows_per_sample=args.flows,
        random_flows=not args.fixed_flows,
        seed=args.seed,
    )
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = GNNCongestionRegressor(args.hidden, args.embedding, args.gnn_model).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch)
            loss = criterion(preds, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    backbone_path = artifacts_dir / args.output
    torch.save(model.encoder.state_dict(), backbone_path)
    print(f"Saved pretrained encoder weights to {backbone_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Supervised pretraining for the GNN congestion encoder.")
    parser.add_argument("--samples", type=int, default=400, help="Number of simulated graphs to generate.")
    parser.add_argument("--flows", type=int, default=6, help="Number of flows per sample when randomizing.")
    parser.add_argument("--fixed-flows", action="store_true", help="Use the fixed flow set instead of random ones.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="PyG DataLoader batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channels inside the GNN.")
    parser.add_argument("--embedding", type=int, default=32, help="Output embedding dimension.")
    parser.add_argument("--gnn-model", choices=SUPPORTED_GNN_MODELS, default="gcn", help="GNN encoder architecture.")
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU even if CUDA is available.")
    parser.add_argument("--output", type=str, default="gnn_pretrained.pt", help="Filename for saved encoder weights.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    train_gnn(args)


if __name__ == "__main__":
    main()
