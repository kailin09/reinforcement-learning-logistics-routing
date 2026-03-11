<h1 align="center">
Reinforcement Learning for Logistics Path Optimization
</h1>

<p align="center">
Q-Learning • Deep Q Network • Dijkstra • A* • Greedy
</p>
Reinforcement Learning for Logistics Path Optimization

---

## 📌 Project Overview

This project explores how **Reinforcement Learning (RL)** can be applied to solve **logistics routing problems** on graph networks.

The objective is to train an intelligent agent that learns efficient transportation routes between logistics hubs.

Implemented approaches include:

- **Q-Learning (Tabular Reinforcement Learning)**
- **Deep Q-Network (DQN)**

These approaches are compared with classical path planning algorithms:

- **Dijkstra**
- **A\***
- **Greedy Search**

---

## 🗺 Logistics Network

<p align="center">
<img src="outputs/dynamic_network_sample.png" width="700">
</p>

The logistics system is modeled as a **graph network**:

- **Nodes** → logistics hubs / warehouses  
- **Edges** → transportation routes  
- **Edge weights** → travel cost or distance  

The reinforcement learning agent learns how to navigate this network efficiently.

---

## Algorithms Implemented

### Q-Learning

Tabular reinforcement learning algorithm that learns routing policies through exploration.

Features:

- Q-table learning
- Epsilon-greedy exploration
- Dynamic traffic environment
- Multi-algorithm performance comparison

---

### Deep Q-Network (DQN)

Deep reinforcement learning method using neural networks.

Key techniques:

- Experience Replay
- Target Network
- Neural Q-value approximation
- PyTorch implementation

---


## 📊 Training Results

Example DQN training results:

Episode 500 Reward: -100

Episode 1000 Reward: 45

Episode 1500 Reward: 47

Episode 2000 Reward: 47

Episode 2500 Reward: 47

Episode 3000 Reward: 47


Testing result:

Path: A -> C -> F -> J -> L

Test Reward: 47

Reached Goal: True


The agent successfully learns an efficient route from source to destination.


---


## 📊 Algorithm Comparison

The project evaluates **Reinforcement Learning (RL)** performance against classical shortest path algorithms.

| Algorithm | Category | Optimality | Avg Cost |
|-----------|----------|------------|----------|
| Q-Learning | Reinforcement Learning | Learned Policy | ~19 |
| Dijkstra | Graph Algorithm | Optimal | ~17 |
| A* | Graph Algorithm | Optimal | ~17 |
| Greedy | Heuristic Search | Not Guaranteed | ~35 |

Dijkstra and A* guarantee **optimal shortest paths**, while reinforcement learning learns **adaptive routing policies through exploration**.

---



## 🗺 Visualization



The project generates visual outputs such as:



\- Training reward curves

\- Smoothed reward plots

\- Network graph visualization

\- Algorithm comparison plots



Example outputs:



\- training_rewards_traffic_state_large_graph.png

\- smoothed_rewards_traffic_state_large_graph.png

\- dynamic_network_sample.png

\- multi_algorithm_cost_comparison.png



---


This repository contains two reinforcement learning implementations for logistics routing:
- Q-Learning (tabular RL)
- Deep Q-Network (DQN)

## Project Structure

```
rl_logistics_path_planning
│
├── q_learning_project
│   ├── env.py
│   ├── q_learning_agent.py
│   ├── main.py
│   └── visualize.py
│
├── dqn_logistics_project
│   ├── main_dqn.py
│   ├── dqn_agent.py
│   ├── env.py
│   └── replay_buffer.py
│
├── outputs
│   ├── training_rewards.png
│   ├── smoothed_rewards.png
│   └── multi_algorithm_cost_comparison.png
│
└── README.md
```

---
## ⚙️ Technologies Used

- Python
- PyTorch
- NumPy
- NetworkX
- Matplotlib

---


## 🚀 How to Run

### Run Q-Learning

python q_learning_agent.py



### Run DQN

cd dqn_logistics_project
python main_dqn.py



---



## Future Improvements

Possible upgrades:

- Double DQN
- Multi-agent routing
- Traffic prediction integration
- Real-world road network using OpenStreetMap



---



## Author

Sim Kai Lin  

Computer Science and Technology  
Beijing Institute of Technology  

Research Interests:

- Reinforcement Learning
- Artificial Intelligent
- Intelligent Transportation Systems
- Logistics Optimization



## Training Results

### Training Reward Curve

<p align="center">
<img src="outputs/training_rewards_traffic_state_large_graph.png" width="700">
</p>

---

### Smoothed Reward Curve

<p align="center">
<img src="outputs/smoothed_rewards_traffic_state_large_graph.png" width="700">
</p>

---

### Algorithm Cost Comparison

<p align="center">
<img src="outputs/multi_algorithm_cost_comparison.png" width="700">
</p>

