from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx


def ensure_output_folder() -> str:
    folder = "outputs"
    os.makedirs(folder, exist_ok=True)
    return folder


def build_nx_graph(graph_data: Dict[str, Dict[str, int]]) -> nx.Graph:
    graph = nx.Graph()
    added_edges = set()

    for node, neighbors in graph_data.items():
        for neighbor, cost in neighbors.items():
            edge = tuple(sorted((node, neighbor)))
            if edge not in added_edges:
                graph.add_edge(node, neighbor, weight=cost)
                added_edges.add(edge)

    return graph


def plot_graph(
    graph_data: Dict[str, Dict[str, int]],
    highlighted_path: Optional[List[str]] = None,
    title: str = "Logistics Network",
    filename: Optional[str] = None
) -> None:
    graph = build_nx_graph(graph_data)
    pos = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(graph, pos, node_size=1300)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")
    nx.draw_networkx_edges(graph, pos, width=2)

    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=12)

    if highlighted_path and len(highlighted_path) > 1:
        path_edges = list(zip(highlighted_path[:-1], highlighted_path[1:]))
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=path_edges,
            width=5
        )

    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.tight_layout()

    if filename:
        folder = ensure_output_folder()
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved graph to: {save_path}")

    plt.show()
    plt.close()


def plot_rewards(rewards: List[float], filename: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Training Reward per Episode", fontsize=16)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()

    if filename:
        folder = ensure_output_folder()
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved reward plot to: {save_path}")

    plt.show()
    plt.close()


def plot_smoothed_rewards(
    rewards: List[float],
    window: int = 20,
    filename: Optional[str] = None
) -> None:
    if len(rewards) < window:
        print("Not enough rewards to smooth.")
        return

    smoothed = []
    for i in range(len(rewards) - window + 1):
        avg = sum(rewards[i:i + window]) / window
        smoothed.append(avg)

    plt.figure(figsize=(10, 5))
    plt.plot(smoothed)
    plt.title(f"Smoothed Training Reward (Window={window})", fontsize=16)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.tight_layout()

    if filename:
        folder = ensure_output_folder()
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved smoothed reward plot to: {save_path}")

    plt.show()
    plt.close()


def plot_cost_comparison(
    rl_costs: List[float],
    dijkstra_costs: List[float],
    filename: Optional[str] = None
) -> None:
    episodes = list(range(1, len(rl_costs) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rl_costs, label="Q-Learning Cost")
    plt.plot(episodes, dijkstra_costs, label="Dijkstra Cost")
    plt.title("RL vs Dijkstra Path Cost Comparison", fontsize=16)
    plt.xlabel("Evaluation Run")
    plt.ylabel("Path Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if filename:
        folder = ensure_output_folder()
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to: {save_path}")

    plt.show()
    plt.close()