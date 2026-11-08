"""
SARSA (State-Action-Reward-State-Action) Policy.
Classic on-policy Temporal-Difference (TD) control algorithm.
Update rule:
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple
import numpy as np

from rl.policies.base import BasePolicy


class SARSAPolicy(BasePolicy):
    """
    Tabular On-Policy SARSA with Epsilon-Greedy Exploration.
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
        super().__init__(n_actions=n_actions, name="SARSA (On-Policy TD)")
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

        self.q_table: Dict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

    def discretize_state(self, observation: np.ndarray) -> Tuple[int, ...]:
        """Convert continuous observation vector to discrete bin indices."""
        bins = np.clip(
            (observation * self.n_bins).astype(int),
            0,
            self.n_bins - 1,
        )
        return tuple(int(b) for b in bins)

    def select_action(self, observation: np.ndarray) -> int:
        """Select action via epsilon-greedy."""
        state = self.discretize_state(observation)
        effective_epsilon = 0.0 if self.eval_mode else self.epsilon

        if self.rng.random() < effective_epsilon:
            return int(self.rng.integers(0, self.n_actions))

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
        On-policy SARSA update.
        Samples next action a' from current epsilon-greedy policy.
        """
        if self.eval_mode:
            return

        state = self.discretize_state(observation)
        next_state = self.discretize_state(next_observation)

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            # On-policy: select next action a' using current policy
            next_action = self.select_action(next_observation)
            next_q = self.q_table[next_state][next_action]
            target = reward + (self.gamma * next_q)

        td_error = target - current_q
        self.q_table[state][action] = current_q + (self.alpha * td_error)

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_eval_mode(self, eval_mode: bool = True) -> None:
        self.eval_mode = eval_mode

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.epsilon = self.initial_epsilon
