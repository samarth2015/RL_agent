import random
import collections
from typing import List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, ...] = (64, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.uint8),
        )


class DeepQAgent:
    """
    Deep Q-Network (DQN) agent.

    Notes:
    - Expects `state` and `next_state` to be array-like (tuple/list/np.ndarray) that can
      be converted to a small float vector (e.g., (bucket_idx, egg_x, egg_y)).
    - `actions` should be a list of discrete actions (typically `list(range(n_actions))`).
    """

    def __init__(
        self,
        actions: List,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_freq: int = 1000,
        device: str = None,
    ):
        self.actions = actions
        self.action_size = len(actions)
        self.name = "Deep Q-Network Agent"
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Default state shape used in the previous code (for compatibility)
        # This agent treats state as a small dense vector: we flatten/convert input.
        self.state_vector_size = 3

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = QNetwork(self.state_vector_size, self.action_size).to(self.device)
        self.target_net = QNetwork(self.state_vector_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr) # Optimizer for policy network
        self.replay = ReplayBuffer(capacity=buffer_size) # Experience replay buffer for storing transitions

        self._train_steps = 0

    def _state_to_vector(self, state) -> np.ndarray:
        # Convert a discrete-state tuple/list to a simple float vector.
        # This keeps the change minimal and compatible with prior discretized states.
        arr = np.array(state, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.array([arr], dtype=np.float32)
        # If length mismatches, try to flatten (robust to small variations)
        if arr.size != self.state_vector_size:
            arr = arr.flatten()[: self.state_vector_size]
            if arr.size < self.state_vector_size:
                # pad with zeros
                pad = np.zeros(self.state_vector_size - arr.size, dtype=np.float32)
                arr = np.concatenate([arr, pad])
        return arr

    def get_action(self, state):
        # Epsilon-greedy: exploration returns an element from self.actions
        if np.random.random() < self.epsilon:
            return random.choice(self.actions)

        state_vec = self._state_to_vector(state)
        with torch.no_grad():
            t = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
            qvals = self.policy_net(t)
            action_idx = int(torch.argmax(qvals, dim=1).cpu().item())
            return action_idx

    def remember(self, state, action, reward, next_state, done):
        # Keep raw states; conversion happens during sampling/training
        self.replay.add(state, action, reward, next_state, done)

    def update(self, state, action, reward, next_state, done):
        """
        Public update method to be called by environment loop.
        Stores experience and performs a training step when possible.
        """
        self.remember(state, action, reward, next_state, done)
        if len(self.replay) >= self.batch_size:
            self._train_step()

    def _train_step(self):
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.from_numpy(np.asarray([self._state_to_vector(s) for s in states], dtype=np.float32)).to(self.device)
        next_states_t = torch.from_numpy(np.asarray([self._state_to_vector(s) for s in next_states], dtype=np.float32)).to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.uint8)).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # Double DQN style target using target network for stability
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)
            max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
            target_q = rewards_t + (1 - dones_t.float()) * (self.gamma * max_next_q)

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping to avoid explosions
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["policy_state"])
        self.target_net.load_state_dict(data.get("target_state", data["policy_state"]))
        if "optimizer" in data:
            try:
                self.optimizer.load_state_dict(data["optimizer"])
            except Exception:
                # optimizer state may be incompatible across versions; ignore safely
                pass