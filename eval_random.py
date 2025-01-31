import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import tyro
import envs as E
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    env_id: str = "Uganda"
    """The environment ID to evaluate."""
    seed: int = 42
    """Random seed for reproducibility."""
    num_episodes: int = 100
    """Number of episodes to evaluate the random policy."""
    track: bool = False
    """If toggled, this evaluation will be tracked with Weights and Biases."""
    wandb_project_name: str = "random_policy_eval"
    """The WandB project name."""
    wandb_entity: Optional[str] = None
    """The WandB entity/team name."""
    max_episode_steps: int = 32
    """The maximum number of steps per episode."""
    wandb_project_name: str = "rulebots"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    gamma: float = 0.95
    """discount factor"""
    agent: str = "random"
    """the agent to evaluate"""  # used for logging
    exp_name: Optional[str] = None
    """the name of the experiment"""


def make_env(env_id, seed, max_episode_steps=None):
    def scale_reward(r):
        return r / max_episode_steps

    def thunk():
        env = gym.make(env_id)
        if env_id == "HeatAlertsNumeric":
            env = gym.wrappers.TransformReward(env, func=scale_reward)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env

    return thunk


if __name__ == "__main__":

    args = tyro.cli(Args)

    run_name = f"random_policy_eval__{args.env_id}__{args.max_episode_steps}__{args.exp_name}__{int(time.time())}"

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Initialize Weights & Biases if tracking is enabled
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create the environment
    env = make_env(args.env_id, args.seed, args.max_episode_steps)()

    # env = E.wrappers.SymlogRewardsWrapper(env)
    env.reset(seed=args.seed)

    logging.info(
        f"Evaluating random policy on {args.env_id} for {args.num_episodes} episodes."
    )

    all_mean_rewards = []
    for episode in tqdm(range(args.num_episodes)):
        obs, info = env.reset()
        done = False

        all_rewards = []
        r = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminations, truncations, info = env.step(action)
            r += reward
            done = terminations or truncations
        all_rewards.append(r)

        # Log results to TensorBoard and WandB
        episodic_return = np.array(all_rewards).mean()
        all_mean_rewards.append(episodic_return)
        writer.add_scalar("charts/episodic_return", episodic_return, episode)

    # Log summary statistics
    mean_return = np.mean(all_mean_rewards)
    std_return = np.std(all_mean_rewards)
    writer.add_scalar("mean_return", mean_return, 0)
    writer.add_scalar("std_return", std_return, 0)

    logging.info(
        f"Evaluation completed: Mean Reward = {mean_return}, Std Reward = {std_return}"
    )

    env.close()
    writer.close()
