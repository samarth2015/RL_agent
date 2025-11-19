import numpy as np
import random

class DoubleQAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.name = "Double Q-Learning Agent"

        # State dimensions based on main.py discretization:
        # Bucket Index (0-10), Egg X (0-10), Egg Y (0-10) + 1 for safety/padding
        self.state_size = (12, 12, 12)
        self.action_size = len(actions)

        # Initialize two Q-tables with zeros
        self.q1 = np.zeros(self.state_size + (self.action_size,))
        self.q2 = np.zeros(self.state_size + (self.action_size,))

    def get_action(self, state):
        """
        Epsilon-greedy policy using the sum of Q1 and Q2.
        """
        # Exploration: Random action
        if np.random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Exploitation: Best action based on Q1 + Q2
        # We add Q1 and Q2 to choose the best action
        q_sum = self.q1[state] + self.q2[state]
        return np.argmax(q_sum)

    def update(self, state, action, reward, next_state, done):
        """
        Double Q-Learning update rule.
        """
        if np.random.random() < 0.5:
            # Update Q1 using Q2 for the value of the next state
            
            # 1. Find the best action for next_state using Q1 (Argmax from Q1)
            best_next_action = np.argmax(self.q1[next_state])
            
            # 2. Calculate Target: Reward + Gamma * Q2(next_state, best_action_from_Q1)
            if done:
                target = reward
            else:
                target = reward + self.gamma * self.q2[next_state][best_next_action]
            
            # 3. Update Q1
            self.q1[state][action] += self.lr * (target - self.q1[state][action])
            
        else:
            # Update Q2 using Q1 for the value of the next state
            
            # 1. Find the best action for next_state using Q2 (Argmax from Q2)
            best_next_action = np.argmax(self.q2[next_state])
            
            # 2. Calculate Target: Reward + Gamma * Q1(next_state, best_action_from_Q2)
            if done:
                target = reward
            else:
                target = reward + self.gamma * self.q1[next_state][best_next_action]
            
            # 3. Update Q2
            self.q2[state][action] += self.lr * (target - self.q2[state][action])