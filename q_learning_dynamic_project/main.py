from __future__ import annotations

import copy
import heapq
from typing import Dict, List, Tuple

from env import DynamicLogisticsEnv
from q_learning_agent import QLearningAgent
from visualize import (
    plot_graph,
    plot_rewards,
    plot_smoothed_rewards,
)


def dijkstra_shortest_path(
    graph: Dict[str, Dict[str, int]],
    start: str,
    goal: str
) -> Tuple[List[str], float]:
    pq: List[Tuple[float, str, List[str]]] = [(0.0, start, [start])]
    visited = set()

    while pq:
        current_cost, current_node, path = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == goal:
            return path, current_cost

        for neighbor, edge_cost in graph[current_node].items():
            if neighbor not in visited:
                heapq.heappush(
                    pq,
                    (current_cost + edge_cost, neighbor, path + [neighbor])
                )

    return [], float("inf")


def heuristic(node: str, goal: str) -> float:
    """
    Simple heuristic for A*.
    Since we don't have real coordinates, we define a rough ranking-based heuristic.
    Smaller means estimated closer to goal L.
    """
    heuristic_values = {
        "A": 10,
        "B": 8,
        "C": 7,
        "D": 7,
        "E": 6,
        "F": 5,
        "G": 4,
        "H": 4,
        "I": 3,
        "J": 2,
        "K": 1,
        "L": 0
    }
    return heuristic_values.get(node, 0)


def a_star_shortest_path(
    graph: Dict[str, Dict[str, int]],
    start: str,
    goal: str
) -> Tuple[List[str], float]:
    """
    A* search on current dynamic graph.
    """
    pq: List[Tuple[float, float, str, List[str]]] = []
    heapq.heappush(pq, (heuristic(start, goal), 0.0, start, [start]))

    visited = {}

    while pq:
        estimated_total, current_cost, current_node, path = heapq.heappop(pq)

        if current_node == goal:
            return path, current_cost

        if current_node in visited and visited[current_node] <= current_cost:
            continue
        visited[current_node] = current_cost

        for neighbor, edge_cost in graph[current_node].items():
            new_cost = current_cost + edge_cost
            priority = new_cost + heuristic(neighbor, goal)
            heapq.heappush(
                pq,
                (priority, new_cost, neighbor, path + [neighbor])
            )

    return [], float("inf")


def greedy_path(
    graph: Dict[str, Dict[str, int]],
    start: str,
    goal: str,
    max_steps: int = 50
) -> Tuple[List[str], float]:
    """
    Improved Greedy baseline:
    - always choose the cheapest unvisited neighbor
    - never revisit a node already in the current path
    - if no valid next node exists, return failure
    """
    current = start
    path = [current]
    total_cost = 0.0
    visited_nodes = {current}
    steps = 0

    while current != goal and steps < max_steps:
        neighbors = graph[current]

        # only consider neighbors not yet visited in current path
        candidates = [
            (cost, neighbor)
            for neighbor, cost in neighbors.items()
            if neighbor not in visited_nodes
        ]

        if not candidates:
            return path, float("inf")

        # choose the lowest-cost unvisited neighbor
        candidates.sort(key=lambda x: x[0])
        best_cost, best_neighbor = candidates[0]

        total_cost += best_cost
        current = best_neighbor
        path.append(current)
        visited_nodes.add(current)
        steps += 1

    if current != goal:
        return path, float("inf")

    return path, total_cost

def train_agent(
    env: DynamicLogisticsEnv,
    agent: QLearningAgent,
    episodes: int = 5000,
    max_steps_per_episode: int = 80
) -> List[float]:
    rewards_per_episode: List[float] = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            actions = env.get_possible_actions(state)
            action = agent.choose_action(state, actions)

            next_state, reward, done = env.step(action)
            next_actions = env.get_possible_actions(next_state) if not done else []

            agent.update(state, action, reward, next_state, next_actions)

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 500 == 0:
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    return rewards_per_episode


def extract_best_path(
    env: DynamicLogisticsEnv,
    agent: QLearningAgent,
    max_steps: int = 40
) -> Tuple[List[str], float]:
    state = env.get_state()
    path = [state[0]]
    total_cost = 0.0
    visited_edges = set()
    steps = 0

    while state[0] != env.goal and steps < max_steps:
        actions = env.get_possible_actions(state)
        if not actions:
            break

        current_node = state[0]
        action = agent.get_best_action(state, actions)
        edge = (current_node, action)

        if edge in visited_edges:
            break

        visited_edges.add(edge)
        total_cost += env.get_edge_cost(current_node, action)

        next_state, _, done = env.step(action)
        path.append(next_state[0])
        state = next_state
        steps += 1

        if done:
            break

    if path[-1] != env.goal:
        return path, float("inf")

    return path, total_cost


def print_single_result(method: str, path: List[str], cost: float) -> None:
    path_str = " -> ".join(path) if path else "No Path"
    cost_str = f"{cost:.2f}" if cost != float("inf") else "INF (failed)"
    print(f"  {method:<12}: {path_str} | Cost = {cost_str}")

def evaluate_agent(
    env: DynamicLogisticsEnv,
    agent: QLearningAgent,
    runs: int = 20
) -> Tuple[List[float], List[float], List[float], List[float]]:
    rl_costs: List[float] = []
    dijkstra_costs: List[float] = []
    a_star_costs: List[float] = []
    greedy_costs: List[float] = []

    print("\nEvaluating under dynamic traffic scenarios...")

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for i in range(runs):
        env.reset()
        graph_snapshot = copy.deepcopy(env.get_current_graph())

        # RL
        env.current_node = env.start
        rl_path, rl_cost = extract_best_path(env, agent)

        # Dijkstra
        dijkstra_path, dijkstra_cost = dijkstra_shortest_path(
            graph_snapshot,
            env.start,
            env.goal
        )

        # A*
        a_star_path, a_star_cost = a_star_shortest_path(
            graph_snapshot,
            env.start,
            env.goal
        )

        # Greedy
        greedy_route, greedy_cost = greedy_path(
            graph_snapshot,
            env.start,
            env.goal
        )

        rl_costs.append(rl_cost)
        dijkstra_costs.append(dijkstra_cost)
        a_star_costs.append(a_star_cost)
        greedy_costs.append(greedy_cost)

        print(f"\nRun {i + 1}:")
        print_single_result("RL", rl_path, rl_cost)
        print_single_result("Dijkstra", dijkstra_path, dijkstra_cost)
        print_single_result("A*", a_star_path, a_star_cost)
        print_single_result("Greedy", greedy_route, greedy_cost)

        if i == 0:
            plot_graph(
                graph_snapshot,
                title="Dynamic Logistics Network (Sample)",
                filename="dynamic_network_sample.png"
            )
            plot_graph(
                graph_snapshot,
                highlighted_path=rl_path,
                title="Q-Learning Dynamic Path (Traffic-Aware State)",
                filename="q_learning_dynamic_path_traffic_state.png"
            )
            plot_graph(
                graph_snapshot,
                highlighted_path=dijkstra_path,
                title="Dijkstra Dynamic Path",
                filename="dijkstra_dynamic_path.png"
            )
            plot_graph(
                graph_snapshot,
                highlighted_path=a_star_path,
                title="A* Dynamic Path",
                filename="a_star_dynamic_path.png"
            )
            plot_graph(
                graph_snapshot,
                highlighted_path=greedy_route,
                title="Greedy Dynamic Path",
                filename="greedy_dynamic_path.png"
            )

    agent.epsilon = original_epsilon
    return rl_costs, dijkstra_costs, a_star_costs, greedy_costs


def summarize_costs(name: str, costs: List[float]) -> None:
    finite_costs = [c for c in costs if c != float("inf")]
    success_rate = f"{len(finite_costs)}/{len(costs)}"

    if not finite_costs:
        print(
            f"{name:<10} | Avg: {'N/A':>6} | Best: {'N/A':>6} | "
            f"Worst: {'N/A':>6} | Success: {success_rate}"
        )
        return

    avg_cost = sum(finite_costs) / len(finite_costs)
    best_cost = min(finite_costs)
    worst_cost = max(finite_costs)

    print(
        f"{name:<10} | Avg: {avg_cost:>6.2f} | "
        f"Best: {best_cost:>6.2f} | Worst: {worst_cost:>6.2f} | "
        f"Success: {success_rate}"
    )

def print_summary(
    rl_costs: List[float],
    dijkstra_costs: List[float],
    a_star_costs: List[float],
    greedy_costs: List[float]
) -> None:
    print("\n===== Evaluation Summary =====")
    summarize_costs("RL", rl_costs)
    summarize_costs("Dijkstra", dijkstra_costs)
    summarize_costs("A*", a_star_costs)
    summarize_costs("Greedy", greedy_costs)

    rl_valid = [c for c in rl_costs if c != float("inf")]
    dijkstra_valid = [c for c in dijkstra_costs if c != float("inf")]

    if rl_valid and dijkstra_valid:
        avg_rl = sum(rl_valid) / len(rl_valid)
        avg_dijkstra = sum(dijkstra_valid) / len(dijkstra_valid)
        print(f"\nAverage Gap (RL - Dijkstra): {avg_rl - avg_dijkstra:.2f}")

        rl_better_equal = sum(
            1 for r, d in zip(rl_costs, dijkstra_costs)
            if r != float("inf") and d != float("inf") and r <= d
        )
        print(f"RL better/equal than Dijkstra: {rl_better_equal}/{len(rl_costs)}")

def save_comparison_table(
    rl_costs: List[float],
    dijkstra_costs: List[float],
    a_star_costs: List[float],
    greedy_costs: List[float]
) -> None:
    import os

    os.makedirs("outputs", exist_ok=True)
    file_path = os.path.join("outputs", "algorithm_comparison_results.csv")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Run,RL,Dijkstra,AStar,Greedy\n")
        for i, (r, d, a, g) in enumerate(
            zip(rl_costs, dijkstra_costs, a_star_costs, greedy_costs),
            start=1
        ):
            r_str = "INF" if r == float("inf") else f"{r:.2f}"
            d_str = "INF" if d == float("inf") else f"{d:.2f}"
            a_str = "INF" if a == float("inf") else f"{a:.2f}"
            g_str = "INF" if g == float("inf") else f"{g:.2f}"
            f.write(f"{i},{r_str},{d_str},{a_str},{g_str}\n")

    print(f"Saved comparison table to: {file_path}")

def plot_multi_algorithm_costs(
    rl_costs: List[float],
    dijkstra_costs: List[float],
    a_star_costs: List[float],
    greedy_costs: List[float]
) -> None:
    import os
    import matplotlib.pyplot as plt

    os.makedirs("outputs", exist_ok=True)
    episodes = list(range(1, len(rl_costs) + 1))

    plt.figure(figsize=(11, 6))
    plt.plot(episodes, rl_costs, label="Q-Learning")
    plt.plot(episodes, dijkstra_costs, label="Dijkstra")
    plt.plot(episodes, a_star_costs, label="A*")
    plt.plot(episodes, greedy_costs, label="Greedy")
    plt.title("Multi-Algorithm Path Cost Comparison", fontsize=16)
    plt.xlabel("Evaluation Run")
    plt.ylabel("Path Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join("outputs", "multi_algorithm_cost_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to: {save_path}")
    plt.show()
    plt.close()


def main() -> None:
    env = DynamicLogisticsEnv()

    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    print("Training traffic-aware Q-Learning agent...")
    rewards = train_agent(env, agent, episodes=5000)

    agent.print_q_table(max_states=20)

    plot_rewards(rewards, filename="training_rewards_traffic_state_large_graph.png")
    plot_smoothed_rewards(
        rewards,
        window=50,
        filename="smoothed_rewards_traffic_state_large_graph.png"
    )

    rl_costs, dijkstra_costs, a_star_costs, greedy_costs = evaluate_agent(env, agent, runs=20)

    print_summary(rl_costs, dijkstra_costs, a_star_costs, greedy_costs)

    save_comparison_table(rl_costs, dijkstra_costs, a_star_costs, greedy_costs)
    plot_multi_algorithm_costs(rl_costs, dijkstra_costs, a_star_costs, greedy_costs)


if __name__ == "__main__":
    main()