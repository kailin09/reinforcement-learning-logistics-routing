import matplotlib.pyplot as plt

from env import DynamicLogisticsEnv
from double_dqn_agent import DQNAgent


env = DynamicLogisticsEnv()
agent = DQNAgent(env.state_dim, len(env.nodes))

episodes = 3000
rewards = []

for episode in range(episodes):

    state = env.reset()
    total_reward = 0

    for step in range(100):

        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)

        next_state, reward, done = env.step(action)

        agent.buffer.push(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

        if done:
            break

    agent.decay_epsilon()

    if episode % 20 == 0:
        agent.update_target()

    rewards.append(total_reward)

    if (episode + 1) % 500 == 0:
        print("Episode", episode + 1, "Reward:", total_reward)


# ===== training finished =====
plt.plot(rewards)
plt.title("DQN Training Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()


# ===== testing starts here =====
print("\nTesting trained Double DQN...")

state = env.reset()
agent.epsilon = 0.0

path = [env.current_node]
total_reward = 0

visited = {env.current_node}

for step in range(20):

    valid_actions = env.get_valid_actions()

    filtered = []

    for a in valid_actions:
        node = env.idx_to_node[a]
        if node not in visited:
            filtered.append(a)

    if filtered:
        action = agent.select_action(state, filtered)
    else:
        action = agent.select_action(state, valid_actions)

    next_state, reward, done = env.step(action)

    path.append(env.current_node)
    visited.add(env.current_node)

    total_reward += reward
    state = next_state

    if done:
        break

print("Path:", " -> ".join(path))
print("Test Reward:", total_reward)
print("Reached Goal:", env.current_node == env.goal)