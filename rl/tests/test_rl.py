"""
Automated unit tests for Gymnasium StudentEnv, baseline policies, and Contextual Bandit.
"""

import pytest
import numpy as np

from rl.envs.student_env import StudentEnv
from rl.policies.random_policy import RandomPolicy
from rl.policies.linear_curriculum import LinearCurriculumPolicy
from rl.policies.contextual_bandit import ContextualBanditPolicy
from rl.evaluation.evaluate import evaluate_policy, run_comparison_experiment


class TestStudentEnv:
    """Tests for the Gymnasium StudentEnv environment."""

    def test_environment_initialization(self):
        env = StudentEnv(n_topics=5, max_steps=20)
        assert env.observation_space.shape == (5,)
        assert env.action_space.n == 5

    def test_reset_conformance(self):
        env = StudentEnv(n_topics=4)
        obs, info = env.reset(seed=123)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert "average_mastery" in info
        assert "step" in info

    def test_reset_reproducibility(self):
        env = StudentEnv(n_topics=4)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_step_contract(self):
        env = StudentEnv(n_topics=3, max_steps=5)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "knowledge_gain" in info
        assert "topic_studied" in info

    def test_invalid_action_raises(self):
        env = StudentEnv(n_topics=3)
        env.reset()
        with pytest.raises(ValueError):
            env.step(99)

    def test_prerequisite_penalty(self):
        # When attempting topic 2 before topic 1 is mastered, should flag prerequisite violation
        env = StudentEnv(n_topics=4, prerequisite_threshold=0.8)
        env.reset(seed=1)
        # Topic 0 has low mastery, attempting topic 2 should violate prerequisite
        _, _, _, _, info = env.step(2)
        assert info["prerequisite_violation"] is True


class TestPolicies:
    """Tests for baseline policies and Contextual Bandit."""

    def test_random_policy(self):
        policy = RandomPolicy(n_actions=4, seed=42)
        obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        for _ in range(20):
            action = policy.select_action(obs)
            assert 0 <= action < 4

    def test_linear_curriculum_advancement(self):
        policy = LinearCurriculumPolicy(n_actions=3, mastery_threshold=0.6)

        # Topic 0 is unmastered: should recommend 0
        obs_low = np.array([0.2, 0.1, 0.1], dtype=np.float32)
        assert policy.select_action(obs_low) == 0

        # Topic 0 is mastered, topic 1 is unmastered: should advance to 1
        obs_advanced = np.array([0.7, 0.3, 0.1], dtype=np.float32)
        assert policy.select_action(obs_advanced) == 1

    def test_contextual_bandit_update(self):
        n_actions = 3
        bandit = ContextualBanditPolicy(
            n_actions=n_actions, context_dim=n_actions, alpha=0.5, seed=42
        )
        obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Initial A should be identity
        initial_trace = np.trace(bandit.A[0])
        assert initial_trace == float(n_actions)

        # Update action 0 with positive reward
        bandit.update(
            observation=obs,
            action=0,
            reward=2.5,
            next_observation=obs,
            done=False,
        )

        # Matrix A and vector b should have changed
        assert np.trace(bandit.A[0]) > initial_trace
        assert np.sum(bandit.b[0]) > 0.0

    def test_contextual_bandit_dimension_check(self):
        bandit = ContextualBanditPolicy(n_actions=3, context_dim=3)
        wrong_dim_obs = np.array([0.1, 0.2], dtype=np.float32)
        with pytest.raises(ValueError):
            bandit.select_action(wrong_dim_obs)


class TestEvaluation:
    """Tests for evaluation runners."""

    def test_evaluate_policy_returns_expected_keys(self):
        env = StudentEnv(n_topics=3, max_steps=5)
        policy = RandomPolicy(n_actions=3, seed=1)
        res = evaluate_policy(policy, env, n_episodes=3, base_seed=10)

        assert res["policy_name"] == "Random Baseline"
        assert res["n_episodes"] == 3
        assert "mean_reward" in res
        assert "mean_mastery" in res
        assert "mean_prereq_violations" in res

    def test_run_comparison_experiment(self):
        results = run_comparison_experiment(
            n_topics=3, max_steps=5, n_episodes=2, base_seed=42
        )
        assert len(results) == 3
        names = [r["policy_name"] for r in results]
        assert "Random Baseline" in names
        assert "Linear Curriculum Baseline" in names
        assert "Contextual Bandit (LinUCB)" in names
