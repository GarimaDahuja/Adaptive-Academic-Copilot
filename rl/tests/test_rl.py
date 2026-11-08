"""
Comprehensive automated unit tests for Gymnasium StudentEnv, baseline policies,
Contextual Bandit, Q-Learning, SARSA, and the Recommendation Service.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from rl.envs.student_env import StudentEnv
from rl.policies.random_policy import RandomPolicy
from rl.policies.linear_curriculum import LinearCurriculumPolicy
from rl.policies.contextual_bandit import ContextualBanditPolicy
from rl.policies.q_learning import QLearningPolicy
from rl.policies.sarsa import SARSAPolicy
from rl.service import AdaptiveRecommendationService
from rl.evaluation.evaluate import evaluate_policy, train_td_agent, run_comparison_experiment


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
        env = StudentEnv(n_topics=4, prerequisite_threshold=0.8)
        env.reset(seed=1)
        _, _, _, _, info = env.step(2)
        assert info["prerequisite_violation"] is True


class TestPolicies:
    """Tests for baselines, contextual bandit, Q-learning, and SARSA."""

    def test_random_policy(self):
        policy = RandomPolicy(n_actions=4, seed=42)
        obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        for _ in range(20):
            action = policy.select_action(obs)
            assert 0 <= action < 4

    def test_linear_curriculum_advancement(self):
        policy = LinearCurriculumPolicy(n_actions=3, mastery_threshold=0.6)

        obs_low = np.array([0.2, 0.1, 0.1], dtype=np.float32)
        assert policy.select_action(obs_low) == 0

        obs_advanced = np.array([0.7, 0.3, 0.1], dtype=np.float32)
        assert policy.select_action(obs_advanced) == 1

    def test_contextual_bandit_update(self):
        n_actions = 3
        bandit = ContextualBanditPolicy(
            n_actions=n_actions, context_dim=n_actions, alpha=0.5, seed=42
        )
        obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        initial_trace = np.trace(bandit.A[0])
        assert initial_trace == float(n_actions)

        bandit.update(
            observation=obs,
            action=0,
            reward=2.5,
            next_observation=obs,
            done=False,
        )

        assert np.trace(bandit.A[0]) > initial_trace
        assert np.sum(bandit.b[0]) > 0.0

    def test_q_learning_discretization_and_update(self):
        q_policy = QLearningPolicy(
            n_actions=3,
            learning_rate=0.5,
            discount_factor=0.9,
            n_bins=5,
            seed=42,
        )
        obs = np.array([0.15, 0.55, 0.85], dtype=np.float32)
        state_key = q_policy.discretize_state(obs)
        assert state_key == (0, 2, 4)

        next_obs = np.array([0.35, 0.55, 0.85], dtype=np.float32)

        # Update action 0
        q_policy.update(
            observation=obs,
            action=0,
            reward=1.0,
            next_observation=next_obs,
            done=False,
        )

        assert q_policy.q_table[state_key][0] > 0.0

    def test_q_learning_eval_mode_is_greedy(self):
        q_policy = QLearningPolicy(n_actions=3, epsilon=1.0, seed=42)
        obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        state_key = q_policy.discretize_state(obs)

        # Make action 2 have the highest Q-value
        q_policy.q_table[state_key][2] = 10.0

        # In eval mode, should deterministically pick action 2
        q_policy.set_eval_mode(True)
        for _ in range(10):
            assert q_policy.select_action(obs) == 2

    def test_sarsa_update(self):
        sarsa_policy = SARSAPolicy(
            n_actions=3,
            learning_rate=0.5,
            discount_factor=0.9,
            seed=42,
        )
        obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        state_key = sarsa_policy.discretize_state(obs)
        next_obs = np.array([0.3, 0.2, 0.3], dtype=np.float32)

        sarsa_policy.update(
            observation=obs,
            action=0,
            reward=2.0,
            next_observation=next_obs,
            done=False,
        )
        assert sarsa_policy.q_table[state_key][0] > 0.0


class TestAdaptiveRecommendationService:
    """Tests for the high-level service and model serialization."""

    def test_recommendation_output(self):
        policy = LinearCurriculumPolicy(n_actions=5)
        service = AdaptiveRecommendationService(policy=policy)

        result = service.recommend([0.1, 0.1, 0.1, 0.1, 0.1])
        assert "recommended_topic_index" in result
        assert "recommended_topic_name" in result
        assert "current_mastery" in result
        assert "rationale" in result
        assert result["recommended_topic_index"] == 0

    def test_q_table_save_and_load_checkpoint(self):
        q_policy = QLearningPolicy(n_actions=3, n_bins=5, seed=42)
        obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        state_key = q_policy.discretize_state(obs)
        q_policy.q_table[state_key][1] = 5.5

        service = AdaptiveRecommendationService(policy=q_policy)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_q_policy.json"
            service.save_policy_checkpoint(ckpt_path)
            assert ckpt_path.exists()

            loaded_service = AdaptiveRecommendationService.load_policy_checkpoint(
                ckpt_path, policy_type="q_learning"
            )
            loaded_rec = loaded_service.recommend([0.1, 0.2, 0.3])
            assert loaded_rec["recommended_topic_index"] == 1


class TestEvaluation:
    """Tests for evaluation and training pipelines."""

    def test_train_td_agent(self):
        env = StudentEnv(n_topics=3, max_steps=5)
        policy = QLearningPolicy(n_actions=3, seed=1)
        rewards = train_td_agent(policy, env, n_train_episodes=5, base_seed=10)
        assert len(rewards) == 5

    def test_run_comparison_experiment(self):
        results = run_comparison_experiment(
            n_topics=3, max_steps=5, n_episodes=2, n_train_episodes=5, base_seed=42
        )
        assert len(results) == 5
        names = [r["policy_name"] for r in results]
        assert "Random Baseline" in names
        assert "Linear Curriculum Baseline" in names
        assert "Contextual Bandit (LinUCB)" in names
        assert "Q-Learning (TD)" in names
        assert "SARSA (On-Policy TD)" in names
