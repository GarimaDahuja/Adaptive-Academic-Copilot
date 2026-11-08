"""
Baseline 2: Linear Curriculum Policy.
Follows a rigid sequential order (Topic 0 -> Topic 1 -> Topic 2...).
Advances to the next topic only after current topic exceeds mastery threshold.
Serves as the standard traditional heuristic benchmark.
"""

from typing import Optional
import numpy as np
from rl.policies.base import BasePolicy


class LinearCurriculumPolicy(BasePolicy):
    """Sequential syllabus progression policy."""

    def __init__(
        self,
        n_actions: int,
        mastery_threshold: float = 0.70,
    ):
        super().__init__(n_actions=n_actions, name="Linear Curriculum Baseline")
        self.mastery_threshold = mastery_threshold
        self.current_topic = 0

    def select_action(self, observation: np.ndarray) -> int:
        """
        Advance linearly through the topics.
        If current topic is sufficiently mastered, move to the next.
        """
        # Find the earliest topic below the threshold
        for topic_idx in range(self.n_actions):
            if observation[topic_idx] < self.mastery_threshold:
                return topic_idx

        # If all are above threshold, focus on the lowest mastery topic
        return int(np.argmin(observation))

    def reset(self, seed: Optional[int] = None) -> None:
        self.current_topic = 0
