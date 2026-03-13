import copy
import random
import numpy as np


class DynamicLogisticsEnv:

    def __init__(self):

        self.base_graph = {
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

        self.graph = copy.deepcopy(self.base_graph)

        self.nodes = sorted(self.base_graph.keys())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}

        self.start = "A"
        self.goal = "L"

        self.current_node = self.start

        self.state_dim = len(self.nodes)

    def reset(self):

        self.graph = copy.deepcopy(self.base_graph)
        self.current_node = self.start

        return self.get_state()

    def get_state(self):

        state = np.zeros(self.state_dim)
        state[self.node_to_idx[self.current_node]] = 1

        return state

    def get_valid_actions(self):

        return [self.node_to_idx[n] for n in self.graph[self.current_node]]

    def step(self, action_idx):

        next_node = self.idx_to_node[action_idx]

        # invalid move
        if next_node not in self.graph[self.current_node]:
            return self.get_state(), -20, False

        cost = self.graph[self.current_node][next_node]

        self.current_node = next_node

        # reach goal
        if self.current_node == self.goal:
            return self.get_state(), 100, True

        # normal move
        return self.get_state(), -2, False