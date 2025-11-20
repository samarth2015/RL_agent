import numpy as np
import random

class SarsaAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.name = "SARSA Agent"

        # State dimensions based on main.py discretization:
        # Bucket Index (0-10), Egg X (0-10), Egg Y (0-10) + 1 for safety/padding
        self.state_size = (12, 12, 12)
        self.action_size = len(actions)

        # Initialize Q-table with zeros
        self.q_table = np.zeros(self.state_size + (self.action_size,))

    def get_action(self, state):
        """
        Epsilon-greedy policy to choose an action.
        """
        # Exploration: Random action
        if np.random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Exploitation: Best action based on current Q-table
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        SARSA update rule: Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        """
        # 1. Choose the next action A' using the same policy (Epsilon-Greedy)
        # Note: In strict SARSA, this A' should be the one actually executed in the next step.
        # Given the main.py loop structure, we calculate it here for the TD target.
        next_action = self.get_action(next_state)

        # 2. Calculate the Target Value
        if done:
            target = reward
        else:
            # SARSA uses the Q-value of the actual next action chosen (On-Policy)
            target = reward + self.gamma * self.q_table[next_state][next_action]

        # 3. Update Q-value for the current state-action pair
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.lr * (target - current_q)