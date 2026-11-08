"""
Reproducible evaluation script comparing RL and non-RL recommendation policies.
Runs policies through identical student learning environments across fixed seeds.
"""

from typing import Dict, List, Optional
import numpy as np

from rl.envs.student_env import StudentEnv
from rl.policies.base import BasePolicy
from rl.policies.random_policy import RandomPolicy
from rl.policies.linear_curriculum import LinearCurriculumPolicy
from rl.policies.contextual_bandit import ContextualBanditPolicy


def evaluate_policy(
    policy: BasePolicy,
    env: StudentEnv,
    n_episodes: int = 50,
    base_seed: int = 42,
    train: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a policy over multiple episodes.

    Args:
        policy: The policy to evaluate.
        env: The StudentEnv instance.
        n_episodes: Number of simulated student trajectories.
        base_seed: Seed for reproducibility.
        train: Whether the policy is allowed to learn (update) during evaluation.

    Returns:
        Dictionary of mean and std metrics across episodes.
    """
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
    base_seed: int = 42,
) -> List[Dict[str, float]]:
    """
    Run an experiment comparing all policies under identical conditions.
    """
    env = StudentEnv(n_topics=n_topics, max_steps=max_steps)

    policies = [
        RandomPolicy(n_actions=n_topics, seed=base_seed),
        LinearCurriculumPolicy(n_actions=n_topics),
        ContextualBanditPolicy(
            n_actions=n_topics,
            context_dim=n_topics,
            alpha=1.0,
            seed=base_seed,
        ),
    ]

    results = []
    for pol in policies:
        pol.reset(seed=base_seed)
        res = evaluate_policy(
            policy=pol,
            env=env,
            n_episodes=n_episodes,
            base_seed=base_seed,
            train=True,
        )
        results.append(res)

    return results


def print_comparison_table(results: List[Dict[str, float]]) -> None:
    """Pretty-print evaluation results as a formatted table."""
    print("=" * 78)
    print("REINFORCEMENT LEARNING & BASELINE EVALUATION REPORT")
    print("=" * 78)
    print(
        f"{'Policy':<30} | {'Mean Reward':<14} | {'Mean Mastery':<12} | {'Prereq Violations':<15}"
    )
    print("-" * 78)
    for r in results:
        reward_str = f"{r['mean_reward']:.2f} +/- {r['std_reward']:.2f}"
        mastery_str = f"{r['mean_mastery'] * 100:.1f}%"
        violations_str = f"{r['mean_prereq_violations']:.1f}"
        print(
            f"{r['policy_name']:<30} | {reward_str:<14} | {mastery_str:<12} | {violations_str:<15}"
        )
    print("=" * 78)


if __name__ == "__main__":
    results = run_comparison_experiment()
    print_comparison_table(results)
