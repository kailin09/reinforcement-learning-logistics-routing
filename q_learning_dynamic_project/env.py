from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple


class DynamicLogisticsEnv:
    def __init__(self) -> None:
        """
        Larger dynamic logistics environment with 12 nodes.
        State includes:
        - current node
        - local traffic pattern of outgoing edges
        """

        self.base_graph: Dict[str, Dict[str, int]] = {
            "A": {"B": 2, "C": 4},
            "B": {"A": 2, "D": 3, "E": 6},
            "C": {"A": 4, "F": 5, "G": 3},
            "D": {"B": 3, "H": 4},
            "E": {"B": 6, "H": 2, "I": 5},
            "F": {"C": 5, "I": 3, "J": 6},
            "G": {"C": 3, "J": 4},
            "H": {"D": 4, "E": 2, "K": 5},
            "I": {"E": 5, "F": 3, "K": 3},
            "J": {"F": 6, "G": 4, "L": 4},
            "K": {"H": 5, "I": 3, "L": 2},
            "L": {"J": 4, "K": 2}
        }

        self.graph: Dict[str, Dict[str, int]] = copy.deepcopy(self.base_graph)

        self.start = "A"
        self.goal = "L"
        self.current_node = self.start

        # dynamic traffic settings
        self.traffic_change_prob = 0.3
        self.traffic_extra_cost_min = 1
        self.traffic_extra_cost_max = 4

    def reset(self) -> Tuple[str, Tuple[str, ...]]:
        self.graph = copy.deepcopy(self.base_graph)
        self._apply_dynamic_traffic()
        self.current_node = self.start
        return self.get_state()

    def _apply_dynamic_traffic(self) -> None:
        """
        Randomly increase edge costs to simulate congestion.
        Keep undirected edges consistent.
        """
        visited_edges = set()

        for node in self.graph:
            for neighbor in self.graph[node]:
                edge = tuple(sorted((node, neighbor)))
                if edge in visited_edges:
                    continue

                visited_edges.add(edge)

                if random.random() < self.traffic_change_prob:
                    extra_cost = random.randint(
                        self.traffic_extra_cost_min,
                        self.traffic_extra_cost_max
                    )
                    self.graph[node][neighbor] += extra_cost
                    self.graph[neighbor][node] += extra_cost

    def get_state(self) -> Tuple[str, Tuple[str, ...]]:
        """
        State = (current_node, local_traffic_pattern)

        local_traffic_pattern compares current edge cost to base edge cost:
        - 'N' = normal
        - 'C' = congested
        """
        neighbors = sorted(self.graph[self.current_node].keys())
        traffic_pattern = []

        for neighbor in neighbors:
            current_cost = self.graph[self.current_node][neighbor]
            base_cost = self.base_graph[self.current_node][neighbor]

            if current_cost > base_cost:
                traffic_pattern.append("C")
            else:
                traffic_pattern.append("N")

        return (self.current_node, tuple(traffic_pattern))

    def get_possible_actions(self, state: Tuple[str, Tuple[str, ...]]) -> List[str]:
        node = state[0]
        return list(self.graph[node].keys())

    def step(self, action: str) -> Tuple[Tuple[str, Tuple[str, ...]], float, bool]:
        if action not in self.graph[self.current_node]:
            return self.get_state(), -30.0, False

        travel_cost = self.graph[self.current_node][action]
        self.current_node = action
        next_state = self.get_state()

        if self.current_node == self.goal:
            reward = 150.0 - float(travel_cost)
            return next_state, reward, True

        reward = -float(travel_cost)
        return next_state, reward, False

    def get_edge_cost(self, from_node: str, to_node: str) -> int:
        return self.graph[from_node][to_node]

    def get_current_graph(self) -> Dict[str, Dict[str, int]]:
        return self.graph