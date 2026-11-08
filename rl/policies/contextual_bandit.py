"""
Contextual Bandit Policy using LinUCB (Linear Upper Confidence Bound).
Selects actions based on the student's context (current topic mastery vector).
Balances exploration of under-tested topics and exploitation of high-reward topics.
"""

from typing import Optional
import numpy as np
from rl.policies.base import BasePolicy


class ContextualBanditPolicy(BasePolicy):
    """
    LinUCB (Disjoint Linear Upper Confidence Bound) Contextual Bandit.

    For each arm (topic), maintains:
        A_a: d x d matrix (initialized to Identity)
        b_a: d-dimensional vector (initialized to zeros)
    Estimated parameter vector:
        theta_a = A_a^(-1) * b_a
    Predicted reward with upper confidence bound:
        p_a = theta_a^T * x + alpha * sqrt(x^T * A_a^(-1) * x)
    """

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        alpha: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(n_actions=n_actions, name="Contextual Bandit (LinUCB)")
        self.context_dim = context_dim
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initialize matrices for each arm
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal models to identity matrices and zero vectors."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # For each action a: A_a = Identity(d, d), b_a = zeros(d)
        self.A = [
            np.identity(self.context_dim, dtype=np.float64)
            for _ in range(self.n_actions)
        ]
        self.b = [
            np.zeros((self.context_dim, 1), dtype=np.float64)
            for _ in range(self.n_actions)
        ]

    def _prepare_context(self, observation: np.ndarray) -> np.ndarray:
        """Ensure context is a column vector of shape (context_dim, 1)."""
        x = np.asarray(observation, dtype=np.float64).reshape(-1, 1)
        if x.shape[0] != self.context_dim:
            raise ValueError(
                f"Expected context dimension {self.context_dim}, got {x.shape[0]}"
            )
        return x

    def select_action(self, observation: np.ndarray) -> int:
        """
        Compute UCB score for each arm given the student's mastery vector.
        Select arm with maximum score. Ties broken randomly using seed.
        """
        x = self._prepare_context(observation)
        p = np.zeros(self.n_actions, dtype=np.float64)

        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]

            # Expected reward estimate
            expected_reward = float((theta.T @ x).item())

            # Exploration bonus (uncertainty bound)
            variance = float((x.T @ A_inv @ x).item())
            # Clip negative numerical precision artifacts
            confidence = self.alpha * np.sqrt(max(0.0, variance))

            p[a] = expected_reward + confidence

        # In case of ties, add a tiny jitter from rng
        jitter = self.rng.uniform(0, 1e-6, size=self.n_actions)
        return int(np.argmax(p + jitter))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update the model for the chosen action using the observed reward.
        A_a <- A_a + x * x^T
        b_a <- b_a + reward * x
        """
        x = self._prepare_context(observation)
        self.A[action] += x @ x.T
        self.b[action] += reward * x
