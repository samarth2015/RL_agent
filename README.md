# 🐣 Reinforcement Learning for Chicken & Eggs Game

This project applies Reinforcement Learning (RL) to train an autonomous agent to play an arcade-style game called **Chicken & Eggs**, where eggs fall from the top of the screen and the agent controls a bucket to catch them.  
Three RL algorithms are implemented and evaluated:

- **SARSA (On-Policy TD Control)**
- **Double Q-Learning**
- **Deep Q-Network (DQN)**

The goal is to compare learning performance, convergence speed, and stability across tabular and deep RL methods.

---

## 🕹️ Game Environment

The game is developed using **Pygame**.

- Chickens remain fixed at the top of the screen and randomly drop eggs.
- The agent controls a **bucket** that moves left/right.
- The objective is to catch eggs and avoid misses.

### 🎯 Actions
| Action ID | Meaning       |
|----------:|---------------|
| 0         | Stay          |
| 1         | Move Left     |
| 2         | Move Right    |

### 📌 Reward Function
| Situation       | Reward |
|-----------------|--------|
| Catch Egg       | +10    |
| Miss Egg        | −50    |
| Game Over (3 misses) | −200 |

---

## 🧠 Reinforcement Learning Algorithms

### 🔹 SARSA
- On-policy Temporal Difference Control
- Learns conservatively and safely, but slower convergence

### 🔹 Double Q-Learning
- Uses two Q-tables to reduce overestimation bias
- Achieves high peak performance but is highly unstable

### 🔹 Deep Q-Network (DQN)
- Neural network approximates Q-values
- Replay buffer and target network for stability
- Most stable learning curve among all methods

---

## 📦 Project Structure

    RL_agent/
    ├── main.py                 # Game environment, step loop & learning episodes
    ├── sarsa.py                # SARSA reinforcement learning agent
    ├── double_q.py             # Double Q-Learning reinforcement learning agent
    ├── deep_q.py               # Deep Q-Network (PyTorch) RL agent
    ├── plots/                  # Performance graphs generated after training
    │   ├── Figure_DeepQN.png   # Learning curve of the DQN agent
    │   ├── Figure_DoubleQ1.png # First Double Q-Learning result (200 episodes)
    │   ├── Figure_DoubleQ2.png # Extended Double Q-Learning result (~320 episodes)
    │   └── Figure_SARSA.png    # Learning curve of the SARSA agent
    ├── README.md               # Project documentation
    └── __pycache__/            # Auto-generated compiled files


---

## 🚀 How to Run

### 1️⃣ Install Dependencies

Run the following command:

    pip install pygame numpy torch matplotlib

### 2️⃣ Choose Algorithm

Open `main.py` and set:

    ALGORITHM_TO_RUN = "sarsa"
    # options: "sarsa", "double_q", "deep_q"

### 3️⃣ Start Training

Run:

    python main.py

> 💡 Hint: you can disable rendering in the code to speed up training during long runs.

---

## 📊 Results Summary

| Algorithm             | Convergence Speed     | Stability | Max Score | Comment                                      |
|-----------------------|----------------------|-----------|-----------|----------------------------------------------|
| SARSA                 | Slow                 | Medium    | ~400      | Safe, on-policy learning                     |
| Double Q-Learning     | Fast after ~150 eps  | Poor      | ~750      | Highest peak but very unstable               |
| Deep Q-Network (DQN)  | Moderate             | High    | ~140      | Most stable and smooth learning curve        |

- DQN shows the **smoothest** and most stable learning.
- Double Q-Learning achieves the **highest peak performance** but with large fluctuations.
- SARSA learns **slowly but consistently**, with fewer collapses.

---

## 🔮 Future Enhancements

- Prioritized experience replay for DQN  
- Curriculum learning with increasing egg drop speed  
- Multi-agent competition (multiple buckets)  
- Using PPO or SAC for continuous control  
- Converting the environment to a Gym-compatible API for benchmarking

---

