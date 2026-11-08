"""
RL Recommendation Service and Model Persistence.
Provides high-level inference methods and model save/load functions.
Decouples the Flask API routes from raw RL algorithms.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from rl.policies.base import BasePolicy
from rl.policies.q_learning import QLearningPolicy
from rl.policies.sarsa import SARSAPolicy
from rl.policies.contextual_bandit import ContextualBanditPolicy


DEFAULT_TOPIC_NAMES = [
    "Foundations & Prerequisite Concepts",
    "Core Data Structures & Principles",
    "Intermediate Algorithms & Problem Solving",
    "Advanced System Architecture",
    "Comprehensive Applications & PYQ Mastery",
]


class AdaptiveRecommendationService:
    """
    High-level recommendation engine that maps student mastery profiles
    to optimal study recommendations using a trained RL policy.
    """

    def __init__(
        self,
        policy: BasePolicy,
        topic_names: Optional[List[str]] = None,
    ):
        self.policy = policy
        self.n_topics = policy.n_actions
        self.topic_names = topic_names or DEFAULT_TOPIC_NAMES[: self.n_actions_safe]

    @property
    def n_actions_safe(self) -> int:
        return self.policy.n_actions

    def recommend(
        self,
        mastery_vector: List[float],
    ) -> Dict[str, Any]:
        """
        Generate an adaptive topic recommendation given student mastery scores.

        Args:
            mastery_vector: List of float scores in [0.0, 1.0] for each topic.

        Returns:
            Dictionary with recommended topic index, topic name, current mastery,
            and learning rationale.
        """
        obs = np.array(mastery_vector, dtype=np.float32)
        if len(obs) != self.policy.n_actions:
            raise ValueError(
                f"Expected {self.policy.n_actions} topic scores, received {len(obs)}"
            )

        # Set policy to evaluation/greedy mode if supported
        if hasattr(self.policy, "set_eval_mode"):
            self.policy.set_eval_mode(True)

        chosen_action = self.policy.select_action(obs)
        topic_name = (
            self.topic_names[chosen_action]
            if chosen_action < len(self.topic_names)
            else f"Topic {chosen_action}"
        )

        current_score = float(obs[chosen_action])

        # Generate educational rationale
        if current_score < 0.3:
            rationale = "Fundamental gaps identified. Focus on foundational concepts."
        elif current_score < 0.7:
            rationale = "Intermediate practice recommended to build fluency and confidence."
        else:
            rationale = "High mastery achieved. Advance to challenging questions or next module."

        return {
            "recommended_topic_index": int(chosen_action),
            "recommended_topic_name": topic_name,
            "current_mastery": round(current_score, 3),
            "policy_used": self.policy.name,
            "rationale": rationale,
        }

    def save_policy_checkpoint(self, filepath: Path | str) -> None:
        """Serialize policy state to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data: Dict[str, Any] = {
            "policy_name": self.policy.name,
            "n_actions": self.policy.n_actions,
        }

        if isinstance(self.policy, (QLearningPolicy, SARSAPolicy)):
            # Convert tuple keys to string keys for JSON serialization
            serialized_table = {
                ",".join(map(str, state)): q_vals.tolist()
                for state, q_vals in self.policy.q_table.items()
            }
            checkpoint_data["q_table"] = serialized_table
            checkpoint_data["n_bins"] = self.policy.n_bins
            checkpoint_data["alpha"] = self.policy.alpha
            checkpoint_data["gamma"] = self.policy.gamma

        elif isinstance(self.policy, ContextualBanditPolicy):
            checkpoint_data["A"] = [A.tolist() for A in self.policy.A]
            checkpoint_data["b"] = [b.tolist() for b in self.policy.b]
            checkpoint_data["alpha"] = self.policy.alpha

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

    @classmethod
    def load_policy_checkpoint(
        cls,
        filepath: Path | str,
        policy_type: str = "q_learning",
    ) -> "AdaptiveRecommendationService":
        """Load a saved policy checkpoint from JSON."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        n_actions = data["n_actions"]

        if policy_type in ("q_learning", "sarsa"):
            policy_cls = QLearningPolicy if policy_type == "q_learning" else SARSAPolicy
            policy = policy_cls(
                n_actions=n_actions,
                learning_rate=data.get("alpha", 0.1),
                discount_factor=data.get("gamma", 0.95),
                n_bins=data.get("n_bins", 5),
            )
            for state_str, q_vals in data.get("q_table", {}).items():
                state_tuple = tuple(int(x) for x in state_str.split(","))
                policy.q_table[state_tuple] = np.array(q_vals, dtype=np.float64)
            policy.set_eval_mode(True)

        elif policy_type == "contextual_bandit":
            policy = ContextualBanditPolicy(
                n_actions=n_actions,
                context_dim=n_actions,
                alpha=data.get("alpha", 1.0),
            )
            policy.A = [np.array(A, dtype=np.float64) for A in data.get("A", [])]
            policy.b = [np.array(b, dtype=np.float64) for b in data.get("b", [])]

        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        return cls(policy=policy)
