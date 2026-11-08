"""
Baseline 1: Random Policy.
Selects study topics uniformly at random without considering student mastery.
Serves as the lower-bound baseline benchmark.
"""

from typing import Optional
import numpy as np
from rl.policies.base import BasePolicy


class RandomPolicy(BasePolicy):
    """Uniform random recommendation policy."""

    def __init__(self, n_actions: int, seed: Optional[int] = None):
        super().__init__(n_actions=n_actions, name="Random Baseline")
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray) -> int:
        """Select a topic uniformly at random."""
        return int(self.rng.integers(0, self.n_actions))

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
