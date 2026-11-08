"""
Abstract Base Policy for Academic Recommendation Agents.
Defines standard interface for both baseline heuristics and RL agents.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BasePolicy(ABC):
    """Abstract interface that all recommendation policies must implement."""

    def __init__(self, n_actions: int, name: str = "BasePolicy"):
        self.n_actions = n_actions
        self.name = name

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        """
        Given the student's current observation (mastery state),
        choose the next action (topic to study).
        """
        pass

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update policy parameters based on observed transition.
        Default implementation is a no-op (for non-learning baselines).
        """
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal policy state (e.g., at the start of a new evaluation)."""
        pass

