<h1 align="center">
Reinforcement Learning for Logistics Path Optimization
</h1>

<p align="center">
Q-Learning • Deep Q Network • Dijkstra • A* • Greedy
</p>
# \# Reinforcement Learning for Logistics Path Optimization

# 

# \## Project Overview

# 

# This project explores how \*\*Reinforcement Learning (RL)\*\* can be applied to solve \*\*logistics routing problems\*\*.

# 

# The objective is to train an intelligent agent that learns to navigate through a logistics network and find efficient routes between warehouses.

# 

# Two reinforcement learning approaches are implemented:

# 

# \- \*\*Q-Learning (Tabular RL)\*\*

# \- \*\*Deep Q-Network (DQN)\*\*

# 

# These models are compared with classical path planning algorithms:

# 

# \- \*\*Dijkstra\*\*

# \- \*\*A\\\*\*\*

# \- \*\*Greedy Search\*\*

# 

# ---

# 

# \# Logistics Network

# <p align="center">
<img src="outputs/dynamic_network_sample.png" width="700">
</p>

# The logistics system is represented as a \*\*graph network\*\*:

# 

# \- Nodes → logistics hubs / warehouses

# \- Edges → transportation routes

# \- Edge weights → travel cost or distance

# 

# Example network:

A → C → F → J → L

A → B → D → H → K → L



---



## Algorithms Implemented

### Q-Learning

Tabular reinforcement learning algorithm that learns routing policies through exploration.

Features:

- Q-table learning
- Epsilon-greedy exploration
- Dynamic traffic environment

---

### Deep Q-Network (DQN)

Deep reinforcement learning method using neural networks.

Key techniques:

- Experience Replay
- Target Network
- Neural Q-value approximation


---


\# Training Example



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



\# Algorithm Comparison



The project evaluates RL performance against classical shortest path algorithms.



| Algorithm | Avg Cost |

|-----------|----------|

| RL (Q-Learning) | ~19 |

| Dijkstra | ~17 |

| A\* | ~17 |

| Greedy | ~35 |



Dijkstra and A\* guarantee optimal solutions, while RL learns adaptive routing policies through exploration.



---



\# Visualization



The project generates visual outputs such as:



\- Training reward curves

\- Smoothed reward plots

\- Network graph visualization

\- Algorithm comparison plots



Example outputs:



\- training\_rewards.png

\- smoothed\_rewards.png

\- dynamic\_network\_sample.png

\- multi\_algorithm\_cost\_comparison.png



---



\# Project Structure

rl\_logistics\_path\_planning



├── q\_learning\_project

│ ├── environment.py

│ ├── q\_learning\_agent.py

│ ├── evaluation.py

│

├── dqn\_logistics\_project

│ ├── main\_dqn.py

│ ├── dqn\_agent.py

│ ├── replay\_buffer.py

│ ├── env.py

│

├── outputs

│ ├── training\_rewards\_traffic\_state\_large\_graph.png

│ ├── smoothed\_rewards\_traffic\_state\_large\_graph.png

│ ├── multi\_algorithm\_cost\_comparison.png

│

└── README.md

---



\# Technologies Used



\- Python

\- PyTorch

\- NumPy

\- Matplotlib

\- NetworkX



---



\# How to Run



\## Run Q-Learning Version

python q\_learning\_agent.py



\## Run DQN Version

cd dqn\_logistics\_project

python main\_dqn.py



---



\# Future Improvements



Possible future upgrades:



\- Double DQN

\- Prioritized Experience Replay

\- Multi-agent logistics routing

\- Traffic-aware routing

\- Integration with real map data (OpenStreetMap)



---



\# Author



Kai Lin Sim  



Computer Science and Technology  

Beijing Institute of Technology  



Interests:



\- Artificial Intelligence

\- Reinforcement Learning

\- Intelligent Transportation Systems

\- Logistics Optimization



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

