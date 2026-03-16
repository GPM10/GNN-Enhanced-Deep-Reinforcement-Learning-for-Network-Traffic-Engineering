import gymnasium as gym
from gymnasium import spaces
import numpy as np
from network_env import NetworkEnvironment
import networkx as nx

class NetworkRoutingEnv(gym.Env):
    def __init__(self, env: NetworkEnvironment, source, target):
        super().__init__()
        self.network_env = env
        self.source = source
        self.target = target
        self.current_node = source
        self.path = [source]

        # Action space: choose next hop, assume max degree 4 for grid
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right

        # State space: current node (x,y), target (x,y), and utilizations of possible links
        # For simplicity, state as flat vector: [current_x, current_y, target_x, target_y, util_up, util_down, util_left, util_right]
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = self.source
        self.path = [self.source]
        return self._get_obs(), {}

    def _get_obs(self):
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
        return np.array([x/3, y/3, tx/3, ty/3] + utils, dtype=np.float32)  # Normalize positions

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

        self.path.append(next_node)
        self.current_node = next_node

        # Reward: - utilization of the link taken
        util = self.network_env.get_link_utilization(self.current_node, next_node)
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