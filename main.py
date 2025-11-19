import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 1000

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
YELLOW = (255, 215, 0)
GREEN = (0, 200, 0)

# Game Constants
BUCKET_WIDTH = 60
BUCKET_HEIGHT = 40
BUCKET_SPEED = 20  
EGG_SPEED = 10
NUM_CHICKENS = 5
MAX_MISSED_EGGS = 3
# MAX_SCORE = 15

# RL Constants
EPISODES = 200  # Increase this for actual training
EPSILON = 0.1   # Exploration rate (if not handled inside agents)


class ChickenGame:
    def __init__(self, render_mode=True):
        self.render_mode = render_mode
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Chicken & Eggs RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)

        # Chicken positions (Static at top)
        self.chicken_x_positions = np.linspace(50, SCREEN_WIDTH - 50, NUM_CHICKENS)
        self.reset()

    def reset(self):
        """Resets the game state for a new episode."""
        self.bucket_x = SCREEN_WIDTH // 2 - BUCKET_WIDTH // 2
        self.eggs = []  # List of [x, y]
        self.score = 0
        self.missed_eggs = 0
        self.game_over = False
        self.frame_count = 0
        return self.get_state()

    def get_state(self):
        """
        Returns the state representation for the RL Agent.
        
        Format: (Bucket_X_Index, Closest_Egg_X_Index, Closest_Egg_Y_Index)
        We discretize coordinates to make it easier for Tabular methods (SARSA/Q-Table).
        For DQN, you might want to return raw normalized coordinates.
        """
        # Discretize Bucket Position (10 regions)
        bucket_idx = int((self.bucket_x / SCREEN_WIDTH) * 10)
        
        # Find closest egg
        closest_egg_x = -1
        closest_egg_y = -1
        
        if self.eggs:
            # Find egg with max Y (closest to bottom)
            lowest_egg = max(self.eggs, key=lambda e: e[1])
            closest_egg_x = int((lowest_egg[0] / SCREEN_WIDTH) * 10)
            closest_egg_y = int((lowest_egg[1] / SCREEN_HEIGHT) * 10)

        return (bucket_idx, closest_egg_x, closest_egg_y)

    def step(self, action):
        """
        Executes one time step within the environment.
        Action: 0 = Stay, 1 = Left, 2 = Right
        Returns: next_state, reward, done
        """
        reward = 0
        
        # 1. Move Bucket
        if action == 1: # Left
            self.bucket_x -= BUCKET_SPEED
        elif action == 2: # Right
            self.bucket_x += BUCKET_SPEED
        
        # Boundary checks
        self.bucket_x = max(0, min(self.bucket_x, SCREEN_WIDTH - BUCKET_WIDTH))

        # 2. Spawn Eggs (Randomly based on frames)
        # if self.frame_count % 20 == 0:
        # if random.random() < 0.6: # 60% chance to spawn
        if len(self.eggs) == 0:
            spawn_x = random.choice(self.chicken_x_positions)
            self.eggs.append([spawn_x, 50]) # Spawn below chicken

        # 3. Move Eggs & Collision Detection
        for egg in self.eggs[:]:
            egg[1] += EGG_SPEED
            
            # Check Catch
            if (self.bucket_x < egg[0] < self.bucket_x + BUCKET_WIDTH) and \
               (SCREEN_HEIGHT - BUCKET_HEIGHT - 10 < egg[1] < SCREEN_HEIGHT - 10):
                self.score += 1
                reward = 10  # Big reward for catch
                self.eggs.remove(egg)
            
            # Check Miss
            elif egg[1] > SCREEN_HEIGHT:
                self.missed_eggs += 1
                reward = -50 # Penalty for dropping
                self.eggs.remove(egg)

        # 4. Check Termination
        if self.missed_eggs >= MAX_MISSED_EGGS:
            self.game_over = True
            reward = -200 # Penalty for losing
        
        # if self.score >= MAX_SCORE:
        #     self.game_over = True
        #     reward = 200 # Bonus for winning episode

        self.frame_count += 1
        
        if self.render_mode:
            self.render()

        return self.get_state(), reward, self.game_over

    def render(self):
        """Draws the game to the screen."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(WHITE)

        # Draw Chickens (Top)
        for cx in self.chicken_x_positions:
            pygame.draw.circle(self.screen, RED, (int(cx), 30), 15)

        # Draw Eggs
        for egg in self.eggs:
            pygame.draw.ellipse(self.screen, YELLOW, (egg[0]-5, egg[1], 10, 15))

        # Draw Bucket
        pygame.draw.rect(self.screen, BLUE, (self.bucket_x, SCREEN_HEIGHT - BUCKET_HEIGHT, BUCKET_WIDTH, BUCKET_HEIGHT))

        # UI Info
        # score_text = self.font.render(f"Score: {self.score}/{MAX_SCORE}", True, BLACK)
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)

        miss_text = self.font.render(f"Missed: {self.missed_eggs}/{MAX_MISSED_EGGS}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(miss_text, (10, 30))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()


# from sarsa import SarsaAgent
from double_q import DoubleQAgent
# from deep_q import DeepQAgent

# --- MOCK AGENT FOR TESTING  ---
class RandomAgent:
    def __init__(self, actions):
        self.actions = actions
        self.name = "Random Agent"
    
    def get_action(self, state):
        return random.choice(self.actions)
    
    def update(self, state, action, reward, next_state, done):
        pass # Random agent doesn't learn


def run_algorithm(algorithm_name):
    env = ChickenGame(render_mode=True) # Set False to speed up training
    
    # Initialize the correct agent based on selection
    actions = [0, 1, 2] # Stay, Left, Right
    
    agent = None
    
    if algorithm_name == "sarsa":
        print("Initializing SARSA Agent...")
        # agent = SarsaAgent(actions) 
        agent = RandomAgent(actions) # Placeholder
        
    elif algorithm_name == "double_q":
        print("Initializing Double Q-Learning Agent...")
        agent = DoubleQAgent(actions)
        # agent = RandomAgent(actions) # Placeholder
        
    elif algorithm_name == "deep_q":
        print("Initializing Deep Q-Learning Agent...")
        # agent = DeepQAgent(actions)
        agent = RandomAgent(actions) # Placeholder
        
    else:
        print("Unknown algorithm. Defaulting to Random.")
        agent = RandomAgent(actions)

    # Metric Tracking
    scores_per_episode = []
    
    print(f"Starting training for {EPISODES} episodes...")

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 1. Agent chooses action
            action = agent.get_action(state)
            
            # 2. Environment performs action
            next_state, reward, done = env.step(action)
            
            # 3. Agent learns
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
        scores_per_episode.append(env.score) # We track game score (eggs caught)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Score: {env.score}, Missed: {env.missed_eggs}")

    env.close()
    return scores_per_episode, agent.name

def plot_results(scores, algo_name):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label=f"{algo_name} Scores")
    # plt.axhline(y=MAX_SCORE, color='g', linestyle='--', label=f"Win Condition {MAX_SCORE} Eggs")
    plt.xlabel("Episode")
    plt.ylabel("Score (Eggs Caught)")
    plt.title(f"Performance of {algo_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Change this variable to test different files: 
    # Options: "sarsa", "double_q", "deep_q"
    ALGORITHM_TO_RUN = "double_q" 
    
    scores, name = run_algorithm(ALGORITHM_TO_RUN)
    plot_results(scores, name)