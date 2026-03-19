import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any
from network_env import NetworkEnvironment
from gnn_model import GNNEmbedding

class NetworkRoutingEnv(gym.Env):
    def __init__(self, env: NetworkEnvironment, source, target, gnn_embedder: Optional[GNNEmbedding] = None):
        super().__init__()
        self.network_env = env
        self.source = source
        self.target = target
        self.current_node = source
        self.path = [source]
        self.gnn_embedder = gnn_embedder
        self._cached_embeddings: Optional[Dict[Any, np.ndarray]] = None
        self.embedding_dim = gnn_embedder.embedding_dim if gnn_embedder else 0

        # Action space: choose next hop, assume max degree 4 for grid
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right

        # State space: current node (x,y), target (x,y), and utilizations of possible links
        # For simplicity, state as flat vector: [current_x, current_y, target_x, target_y, util_up, util_down, util_left, util_right]
        base_obs_dim = 8
        obs_dim = base_obs_dim + (2 * self.embedding_dim)
        low = np.full((obs_dim,), -10.0 if self.embedding_dim else 0.0, dtype=np.float32)
        high = np.full((obs_dim,), 10.0 if self.embedding_dim else 1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.network_env.reset_loads()
        self.current_node = self.source
        self.path = [self.source]
        self._cached_embeddings = None
        return self._get_obs(), {}

    def _base_features(self):
        x, y = self.current_node
        tx, ty = self.target
        # Possible actions: up, down, left, right
        directions = [(-1,0), (1,0), (0,-1), (0,1)]  # up, down, left, right
        utils = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.network_env.G.has_edge((x,y), (nx,ny)):
                util = self.network_env.get_link_utilization((x,y), (nx,ny))
            else:
                util = 1.0  # High penalty for invalid
            utils.append(util)
        return np.array([x/3, y/3, tx/3, ty/3] + utils, dtype=np.float32)

    def _ensure_embeddings(self):
        if not self.gnn_embedder:
            return
        if self._cached_embeddings is None:
            self._cached_embeddings = self.gnn_embedder.get_embeddings(self.network_env.G)

    def _node_embedding(self, node):
        if not self.gnn_embedder or self.embedding_dim == 0:
            return np.array([], dtype=np.float32)
        self._ensure_embeddings()
        if not self._cached_embeddings:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        embedding = self._cached_embeddings.get(node)
        if embedding is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return embedding.astype(np.float32)

    def _get_obs(self):
        features = self._base_features()
        if self.embedding_dim == 0:
            return features
        current_emb = self._node_embedding(self.current_node)
        target_emb = self._node_embedding(self.target)
        return np.concatenate([features, current_emb, target_emb]).astype(np.float32)

    def step(self, action):
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        dx, dy = directions[action]
        next_node = (self.current_node[0] + dx, self.current_node[1] + dy)

        if not self.network_env.G.has_edge(self.current_node, next_node):
            # Invalid move, penalty
            reward = -10
            done = True
            truncated = False
            return self._get_obs(), reward, done, truncated, {}

        prev_node = self.current_node
        self.network_env.add_traffic([prev_node, next_node], amount=1)
        self.path.append(next_node)
        self.current_node = next_node
        self._cached_embeddings = None

        # Reward: - utilization of the link taken
        util = self.network_env.get_link_utilization(prev_node, next_node)
        reward = -util

        done = (self.current_node == self.target)
        truncated = False

        if done:
            # Bonus for reaching target
            reward += 10

        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        pass

# For training, perhaps a wrapper or separate function.

from stable_baselines3 import PPO

def train_rl_agent(env, total_timesteps=10000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model
