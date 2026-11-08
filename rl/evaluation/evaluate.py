"""
Reproducible evaluation runner comparing all RL and baseline policies:
- Random Baseline
- Linear Curriculum (Traditional heuristic)
- LinUCB Contextual Bandit
- Q-Learning (Off-policy TD)
- SARSA (On-policy TD)
"""

from typing import Dict, List, Optional
import numpy as np

from rl.envs.student_env import StudentEnv
from rl.policies.base import BasePolicy
from rl.policies.random_policy import RandomPolicy
from rl.policies.linear_curriculum import LinearCurriculumPolicy
from rl.policies.contextual_bandit import ContextualBanditPolicy
from rl.policies.q_learning import QLearningPolicy
from rl.policies.sarsa import SARSAPolicy


def train_td_agent(
    policy: BasePolicy,
    env: StudentEnv,
    n_train_episodes: int = 300,
    base_seed: int = 100,
) -> List[float]:
    """
    Train a Temporal-Difference agent (Q-Learning or SARSA) over multiple episodes.
    Decays exploration rate and learns action-value Q-table.
    """
    episode_rewards: List[float] = []

    for ep in range(n_train_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        terminated = False
        truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            action = policy.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            policy.update(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=terminated or truncated,
            )
            ep_reward += reward
            obs = next_obs

        episode_rewards.append(ep_reward)

    return episode_rewards


def evaluate_policy(
    policy: BasePolicy,
    env: StudentEnv,
    n_episodes: int = 50,
    base_seed: int = 42,
    train: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a policy over multiple deterministic episodes without learning updates.
    """
    if hasattr(policy, "set_eval_mode"):
        policy.set_eval_mode(not train)

    total_rewards: List[float] = []
    final_masteries: List[float] = []
    steps_taken: List[int] = []
    prereq_violations: List[int] = []

    for episode in range(n_episodes):
        seed = base_seed + episode
        obs, info = env.reset(seed=seed)
        episode_reward = 0.0
        episode_violations = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy.select_action(obs)
            next_obs, reward, terminated, truncated, step_info = env.step(action)

            if train:
                policy.update(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    done=terminated or truncated,
                )

            if step_info.get("prerequisite_violation", False):
                episode_violations += 1

            episode_reward += reward
            obs = next_obs

        total_rewards.append(episode_reward)
        final_masteries.append(float(np.mean(obs)))
        steps_taken.append(env.current_step)
        prereq_violations.append(episode_violations)

    return {
        "policy_name": policy.name,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_mastery": float(np.mean(final_masteries)),
        "std_mastery": float(np.std(final_masteries)),
        "mean_steps": float(np.mean(steps_taken)),
        "mean_prereq_violations": float(np.mean(prereq_violations)),
    }


def run_comparison_experiment(
    n_topics: int = 5,
    max_steps: int = 30,
    n_episodes: int = 50,
    n_train_episodes: int = 300,
    base_seed: int = 42,
) -> List[Dict[str, float]]:
    """
    Run the full suite of baselines and RL algorithms under identical seeds.
    """
    env = StudentEnv(n_topics=n_topics, max_steps=max_steps)

    # 1. Random Baseline
    random_policy = RandomPolicy(n_actions=n_topics, seed=base_seed)

    # 2. Linear Curriculum Baseline
    curriculum_policy = LinearCurriculumPolicy(n_actions=n_topics)

    # 3. Contextual Bandit (LinUCB) - evaluates while updating in real-time
    bandit_policy = ContextualBanditPolicy(
        n_actions=n_topics,
        context_dim=n_topics,
        alpha=1.0,
        seed=base_seed,
    )

    # 4. Q-Learning (Off-policy TD) - pre-train then evaluate pure greedy
    q_policy = QLearningPolicy(
        n_actions=n_topics,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        seed=base_seed,
    )
    train_td_agent(q_policy, env, n_train_episodes=n_train_episodes, base_seed=1000)

    # 5. SARSA (On-policy TD) - pre-train then evaluate pure greedy
    sarsa_policy = SARSAPolicy(
        n_actions=n_topics,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        seed=base_seed,
    )
    train_td_agent(sarsa_policy, env, n_train_episodes=n_train_episodes, base_seed=1000)

    # Run deterministic evaluations
    results = [
        evaluate_policy(random_policy, env, n_episodes, base_seed, train=False),
        evaluate_policy(curriculum_policy, env, n_episodes, base_seed, train=False),
        evaluate_policy(bandit_policy, env, n_episodes, base_seed, train=True),
        evaluate_policy(q_policy, env, n_episodes, base_seed, train=False),
        evaluate_policy(sarsa_policy, env, n_episodes, base_seed, train=False),
    ]

    return results


def print_comparison_table(results: List[Dict[str, float]]) -> None:
    """Pretty-print evaluation results as a formatted table."""
    print("=" * 82)
    print("REINFORCEMENT LEARNING & BASELINE COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 82)
    print(
        f"{'Policy':<28} | {'Mean Reward':<16} | {'Mean Mastery':<12} | {'Prereq Violations':<17}"
    )
    print("-" * 82)
    for r in results:
        reward_str = f"{r['mean_reward']:.2f} +/- {r['std_reward']:.2f}"
        mastery_str = f"{r['mean_mastery'] * 100:.1f}%"
        violations_str = f"{r['mean_prereq_violations']:.1f}"
        print(
            f"{r['policy_name']:<28} | {reward_str:<16} | {mastery_str:<12} | {violations_str:<17}"
        )
    print("=" * 82)


if __name__ == "__main__":
    results = run_comparison_experiment()
    print_comparison_table(results)
