from __future__ import annotations

import random
from typing import Any, Dict, List


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table now supports dynamic tuple states
        self.q_table: Dict[Any, Dict[str, float]] = {}

    def _ensure_state_exists(self, state: Any) -> None:
        if state not in self.q_table:
            self.q_table[state] = {}

    def _initialize_actions_if_needed(self, state: Any, actions: List[str]) -> None:
        self._ensure_state_exists(state)
        for action in actions:
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0

    def choose_action(self, state: Any, actions: List[str]) -> str:
        if not actions:
            raise ValueError(f"No available actions from state {state}")

        self._initialize_actions_if_needed(state, actions)

        if random.random() < self.epsilon:
            return random.choice(actions)

        return max(actions, key=lambda action: self.q_table[state][action])

    def update(
        self,
        state: Any,
        action: str,
        reward: float,
        next_state: Any,
        next_actions: List[str]
    ) -> None:
        self._initialize_actions_if_needed(state, [action])
        self._initialize_actions_if_needed(next_state, next_actions)

        max_next_q = 0.0
        if next_actions:
            max_next_q = max(self.q_table[next_state][a] for a in next_actions)

        old_q = self.q_table[state][action]
        new_q = old_q + self.alpha * (
            reward + self.gamma * max_next_q - old_q
        )
        self.q_table[state][action] = new_q

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_best_action(self, state: Any, actions: List[str]) -> str:
        if not actions:
            raise ValueError(f"No available actions from state {state}")

        self._initialize_actions_if_needed(state, actions)
        return max(actions, key=lambda action: self.q_table[state][action])

    def print_q_table(self, max_states: int = 20) -> None:
        print("\nQ-Table (sample states):")
        count = 0
        for state, actions in self.q_table.items():
            print(f"{state}:")
            for action, value in actions.items():
                print(f"  -> {action}: {value:.4f}")
            count += 1
            if count >= max_states:
                print("... (truncated)")
                break