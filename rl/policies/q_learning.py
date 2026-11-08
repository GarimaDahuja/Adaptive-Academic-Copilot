"""
Q-Learning (Temporal-Difference Learning) Policy.
Learns action-value function Q(s, a) to estimate cumulative future discounted rewards.
Enables sequential planning so the agent learns prerequisite dependencies over time.
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple
import numpy as np

from rl.policies.base import BasePolicy


class QLearningPolicy(BasePolicy):
    """
    Tabular Q-Learning Agent with State Discretization and Epsilon-Greedy Exploration.

    Bellman Equation update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        n_bins: int = 5,
        seed: Optional[int] = None,
    ):
        super().__init__(n_actions=n_actions, name="Q-Learning (TD)")
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_bins = n_bins
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.eval_mode = False

        # Q-table: maps discretized state tuple -> numpy array of Q-values for each action
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

    def discretize_state(self, observation: np.ndarray) -> Tuple[int, ...]:
        """
        Discretize a continuous mastery vector in [0.0, 1.0]^d into discrete bin indices.
        Example with n_bins=5: [0.12, 0.45] -> (0, 2)
        """
        bins = np.clip(
            (observation * self.n_bins).astype(int),
            0,
            self.n_bins - 1,
        )
        return tuple(int(b) for b in bins)

    def select_action(self, observation: np.ndarray) -> int:
        """
        Select action using epsilon-greedy exploration.
        In eval_mode, epsilon is treated as 0 (pure greedy exploitation).
        """
        state = self.discretize_state(observation)
        effective_epsilon = 0.0 if self.eval_mode else self.epsilon

        # Exploration: random action
        if self.rng.random() < effective_epsilon:
            return int(self.rng.integers(0, self.n_actions))

        # Exploitation: greedy action with tie-breaking
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.where(np.isclose(q_values, max_q))[0]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Perform standard Temporal-Difference (TD) Q-learning update.
        """
        if self.eval_mode:
            return

        state = self.discretize_state(observation)
        next_state = self.discretize_state(next_observation)

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_state])
            target = reward + (self.gamma * best_next_q)

        td_error = target - current_q
        self.q_table[state][action] = current_q + (self.alpha * td_error)

        # Decay exploration rate epsilon towards epsilon_min
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_eval_mode(self, eval_mode: bool = True) -> None:
        """Toggle between training and evaluation (greedy) mode."""
        self.eval_mode = eval_mode

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset RNG and restore initial exploration rate."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.epsilon = self.initial_epsilon

