# Removed unused commented-out code and imports
import logging
import os
import random
import re
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import pandas as pd
from typing import Optional
import warnings

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsummary
import tyro
from langchain_together import TogetherEmbeddings
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.distributions.categorical import Categorical

from agents import LLMRulesAgent
import buffers
from llm_apis import get_llm_api, ValidLLMs
import envs as E

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment."""
    seed: int = 1
    """Seed of the experiment."""
    torch_deterministic: bool = True
    """If toggled, `torch.backends.cudnn.deterministic=True`."""
    cuda: bool = False
    """If toggled, CUDA will be enabled by default."""
    track: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases."""
    wandb_project_name: str = "rulebots"
    """The Weights and Biases project name."""
    wandb_entity: Optional[str] = None
    """The entity (team) of the Weights and Biases project."""
    log_frequency: int = 8
    """The logging frequency of the algorithm."""

    # Environment
    env_id: str = "UgandaNumeric"
    """The ID of the environment."""
    num_envs: int = 4
    """The number of parallel game environments."""
    max_episode_steps: Optional[int] = 32
    """The maximum number of steps per episode."""
    min_temperature_threshold: float = 0.0
    """Only used for the heat alert environment."""

    # Algorithm
    total_timesteps: int = 50000
    """Total timesteps of the experiments."""
    gamma: float = 0.95
    """The discount factor gamma."""
    tau: float = 0.25
    """Target smoothing coefficient."""
    batch_size: int = 64
    """The batch size of samples from the replay memory."""
    learning_starts: int = 256
    """Timestep to start learning."""
    policy_lr: float = 1e-4
    """The learning rate of the policy network optimizer."""
    q_lr: float = 1e-4
    """The learning rate of the Q-network optimizer."""
    update_frequency: float | int = 8
    """The frequency of training updates."""
    warmup_updates: int = 1
    """The number of warmup updates to the value function on the first iteration."""
    actor_updates: int = 1
    """The number of updates to the actor per update cycle."""
    critic_updates: int = 1
    """The number of updates to the critic per update cycle."""
    target_network_frequency: int = 64
    """The frequency of updates for the target networks."""
    alpha: float = 0.01
    """Entropy regularization coefficient."""
    autotune: bool = True
    """Automatic tuning of the entropy coefficient."""
    target_entropy_scale: float = 0.89
    """Coefficient for scaling the autotune entropy target."""
    # reinit: bool = False
    # """If toggled, the experiment will be reinitialized."""

    # Eval
    eval: bool = True
    """If toggled, the agent will be evaluated."""
    eval_interval: int = 1
    """The evaluation interval."""
    rolling_returns_window: int = 16
    """The rolling rewards window."""
    eval_only: bool = False
    """If toggled, skip training and directly evaluate using a pre-trained model."""
    load_model_path: Optional[str] = None
    """Path to the saved actor model to load during evaluation only."""
    # LLM
    num_rules: int = 10
    """The number of rules for the rule-based LLM-only agent."""
    llm: ValidLLMs = "gpt-4o-mini-huit"
    """The language model to use."""
    embedder_lm: str = "togethercomputer/m2-bert-80M-8k-retrieval"
    """The embedding model to use."""
    hidden_dim: int = 16
    """The hidden dimension of the networks."""

    # Buffer collection mode
    load_buffer: bool = False
    """If toggled, the agent will load the buffer from the pickle file if it exists."""
    buffer_size: int = 4096
    """The replay memory buffer size."""

    agent: Optional[str] = "sac_numeric"
    """The agent to use."""
    thoughts: bool = True
    """If toggled, the agent will use thoughts."""

    # Torch compile
    compile_torch: bool = False
    """If toggled, the models will be compiled with Torch."""


def make_env(env_id, seed, max_episode_steps=None, eval=False):
    def thunk():
        env = gym.make(env_id)
        if env_id == "HeatAlertsNumeric":
            env.min_temperature_threshold = args.min_temperature_threshold
        elif env_id in ("UgandaNumeric", "MimicIIINumeric", "MimicIVNumeric"):
            env = gym.wrappers.FlattenObservation(env)
        elif env_id in ("BinPackingNumeric", "BinPackingIncrementalNumeric"):
            pass
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env

    return thunk


def make_env_lang(env_id, seed, max_episode_steps=None, eval=False):
    def thunk():
        env = gym.make(env_id)
        if env_id not in ("BinPacking", "BinPackingIncremental"):
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env

    return thunk


def action_parser(s: str, n: int) -> int:
    """
    Convert the action from text to an integer.
    """
    numbers = re.findall(r"\d+", str(s))
    for num in numbers:
        if int(num) < n:
            return int(num)

        warnings.warn(f"Invalid action: {s}, returning a random action")
        act = random.randint(0, n - 1)
        return int(act)


def update_critic(
    buffer,
    batch_size,
    gamma,
    alpha,
    qf1,
    qf2,
    qf1_target,
    qf2_target,
    q_optimizer,
    actor,
    device,
):
    """
    Update the critic networks (qf1 and qf2) using the sampled data from the replay buffer.

    Args:
        buffer: Replay buffer containing training samples.
        batch_size: Size of the batch to sample.
        gamma: Discount factor for future rewards.
        alpha: Entropy regularization coefficient.
        qf1, qf2: Critic networks to be updated.
        qf1_target, qf2_target: Target critic networks.
        q_optimizer: Optimizer for the critic networks.
        actor: Actor network to be used for policy evaluation.
        device: Device for tensor computations.

    Returns:
        qf_loss: Combined loss for both Q networks.
        qf1_loss: Loss for Q-network 1.
        qf2_loss: Loss for Q-network 2.
    """
    qf1.train()
    qf2.train()

    data = buffer.sample(batch_size)
    next_obs_vec = data["next_obs_vec"]
    rewards = data["rewards"]
    dones = data["dones"]

    with torch.no_grad():
        dist = Categorical(logits=actor(next_obs_vec))
        next_action_probs = F.softmax(dist.logits, dim=-1)
        next_state_log_pi = F.log_softmax(dist.logits, dim=-1)

        qf1_next_tgt = qf1_target(next_obs_vec)
        qf2_next_tgt = qf2_target(next_obs_vec)

        min_qf_next_target = (
            next_action_probs
            * (torch.min(qf1_next_tgt, qf2_next_tgt) - alpha * next_state_log_pi)
        ).sum(-1)
        next_q_value = rewards + (1 - dones) * gamma * min_qf_next_target

    obs_vec = data["obs_vec"]
    sel_idxs = data["actions"].unsqueeze(-1).to(device)

    qf1_values = qf1(obs_vec)
    qf2_values = qf2(obs_vec)

    qf1_a_values = qf1_values.gather(1, sel_idxs).squeeze(-1)
    qf2_a_values = qf2_values.gather(1, sel_idxs).squeeze(-1)

    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    q_optimizer.zero_grad()
    qf_loss.backward()
    q_optimizer.step()

    qf1.eval()
    qf2.eval()

    qss = next_q_value.pow(2).mean().item()

    return (
        qf_loss.item(),
        qf1_loss.item(),
        qf1_a_values,
        qf2_loss.item(),
        qf2_a_values,
        qss,
    )


def update_actor(
    buffer,
    batch_size,
    alpha,
    actor_optimizer,
    qf1,
    qf2,
    actor,
    device,
):
    """
    Update the actor network using the sampled data from the replay buffer.

    Args:
        buffer: Replay buffer containing training samples.
        batch_size: Size of the batch to sample.
        alpha: Entropy regularization coefficient.
        actor_optimizer: Optimizer for the actor network.
        qf1, qf2: Critic networks used for policy evaluation.
        actor: Actor network to be updated.
        device: Device for tensor computations.

    Returns:
        actor_loss: Loss for the actor network.
    """
    actor.train()
    data = buffer.sample(batch_size)
    obs_vec = data["obs_vec"]

    dist = Categorical(logits=actor(obs_vec))
    log_probs = F.log_softmax(dist.logits, dim=-1)
    probs = dist.probs

    with torch.no_grad():
        qf1_values = qf1(obs_vec)
        qf2_values = qf2(obs_vec)

        if qf1_values.is_nested:
            qf1_values = torch.nested.to_padded_tensor(qf1_values, 0.0)
            qf2_values = torch.nested.to_padded_tensor(qf2_values, 0.0)

        min_qf_values = torch.min(qf1_values, qf2_values)

    actor_loss = (probs * (alpha * log_probs - min_qf_values)).mean()
    entropy = dist.entropy().mean().item()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    actor.eval()

    return actor_loss.item(), entropy, probs, log_probs


def update_alpha(
    target_entropy,
    log_alpha,
    alpha_optimizer,
    probs,
    log_probs,
):
    """
    Update the entropy coefficient alpha using the sampled data from the replay buffer.

    Args:
        buffer: Replay buffer containing training samples.
        batch_size: Size of the batch to sample.
        target_entropy: Desired entropy level for the policy.
        log_alpha: Logarithm of the alpha parameter.
        alpha_optimizer: Optimizer for the log_alpha parameter.
        lang_agent: Language agent to calculate policy from embeddings.
        device: Device for tensor computations.

    Returns:
        alpha_loss: Loss for updating alpha.
        alpha: Updated value of alpha.
    """
    # Alpha loss computation
    alpha_loss = (
        probs.detach() * (-log_alpha.exp() * (log_probs + target_entropy).detach())
    ).mean()

    # Update log_alpha
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    # Get the updated alpha value
    alpha = log_alpha.exp().item()

    return alpha_loss.item(), alpha


def main(args: Args):
    run_name = f"{args.env_id}__{args.seed}__{args.exp_name}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            id=run_name,
            resume=False,
            # reinit=args.reinit,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    train_env_funs = [
        make_env(args.env_id + "Numeric", args.seed + i, args.max_episode_steps)
        for i in range(args.num_envs)
    ]
    eval_env_funs = [
        make_env(
            args.env_id + "Numeric",
            1000 * args.seed + i,
            args.max_episode_steps,
            eval=True,
        )
        for i in range(args.num_envs)
    ]
    envs = gym.vector.SyncVectorEnv(train_env_funs)
    eval_envs = gym.vector.SyncVectorEnv(eval_env_funs)

    envs_lang = gym.vector.SyncVectorEnv(
        [
            make_env_lang(
                args.env_id,
                args.seed + i,
                args.max_episode_steps,
            )
            for i in range(args.num_envs)
        ]
    )
    # if args.num_envs == 1:
    #     envs = make_env(args.env_id + "Numeric", args.seed, args.max_episode_steps)()
    #     eval_envs = make_env(
    #         args.env_id + "Numeric", 1000 * args.seed, args.max_episode_steps, eval=True
    #     )()
    # else:
    #     train_env_funs = [
    #         make_env(args.env_id + "Numeric", args.seed + i, args.max_episode_steps)
    #         for i in range(args.num_envs)
    #     ]
    #     eval_env_funs = [
    #         make_env(
    #             args.env_id + "Numeric",
    #             1000 * args.seed + i,
    #             args.max_episode_steps,
    #             eval=True,
    #         )
    #         for i in range(args.num_envs)
    #     ]
    #     envs = gym.vector.SyncVectorEnv(train_env_funs)
    #     eval_envs = gym.vector.SyncVectorEnv(eval_env_funs)
    # if args.num_envs == 1:
    #     envs_lang = make_env_lang(args.env_id, args.seed, args.max_episode_steps)()
    # else:
    #     envs_lang = gym.vector.SyncVectorEnv(
    #         [
    #             make_env_lang(
    #                 args.env_id,
    #                 args.seed + i,
    #                 args.max_episode_steps,
    #             )
    #             for i in range(args.num_envs)
    #         ]
    #     )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    actor = nn.Sequential(
        nn.LayerNorm(envs.single_observation_space.shape[-1]),
        layer_init(nn.Linear(envs.single_observation_space.shape[-1], args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), std=1.0),
    )

    qf1 = nn.Sequential(
        nn.LayerNorm(envs.single_observation_space.shape[-1]),
        layer_init(nn.Linear(envs.single_observation_space.shape[-1], args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), 0.01),
    )

    qf2 = nn.Sequential(
        nn.LayerNorm(envs.single_observation_space.shape[-1]),
        layer_init(nn.Linear(envs.single_observation_space.shape[-1], args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), 0.01),
    )

    actor.eval()
    qf1.eval()
    qf2.eval()

    logging.info("--- Actor ---")
    torchsummary.summary(actor)

    logging.info("--- Q-function ---")
    torchsummary.summary(qf1)

    def critic(rules_emb, obs_vec):
        q1 = qf1(rules_emb, obs_vec)
        q2 = qf2(rules_emb, obs_vec)
        return torch.min(q1, q2)

    q_params = list(qf1.parameters()) + list(qf2.parameters())
    q_optimizer = optim.Adam(q_params, lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr, eps=1e-4)

    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(
            1 / torch.tensor(envs.single_action_space.n)
        )
        log_alpha = torch.scalar_tensor(
            np.log(args.alpha), requires_grad=True, device=device
        )
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    buffer = buffers.SimpleDictReplayBuffer(args.buffer_size, device=device)

    starting_step = 0
    start_time = time.time()
    best_total_reward = -float("inf")
    best_model = None
    best_model_epoch = -1

    qf1_target = deepcopy(qf1)
    qf2_target = deepcopy(qf2)

    if args.compile_torch:
        actor = torch.compile(actor)
        qf1 = torch.compile(qf1)
        qf2 = torch.compile(qf2)
        qf1_target = torch.compile(qf1_target)
        qf2_target = torch.compile(qf2_target)

    obs_vec, _ = envs.reset()
    obs_vec = torch.FloatTensor(obs_vec).to(device)
    autoreset = np.zeros(args.num_envs, dtype=bool)

    _ep_buffer = defaultdict(lambda: [[] for _ in range(args.num_envs)])
    _rolling_returns = deque(maxlen=args.rolling_returns_window)

    _running_qss = 0.0
    _running_qse = 0.0


    # === 👇 load actor if eval_only is on ===
    if args.eval_only:
        assert args.load_model_path is not None, "Must provide --load_model_path in eval_only mode"
        actor.load_state_dict(torch.load(args.load_model_path, map_location=device))
        actor.eval()
        logging.info(f"Loaded actor model from {args.load_model_path}")
    else:
        for global_step in tqdm(range(starting_step, args.total_timesteps)):
            with torch.no_grad():
                action_logits = actor(obs_vec)
                action_dist = Categorical(logits=action_logits)
                actions = action_dist.sample()

            next_obs_vec, env_rewards, dones, trunc, infos = envs.step(actions)
            dones = torch.FloatTensor(dones).to(device)
            next_obs_vec = torch.FloatTensor(next_obs_vec).to(device)

            env_rewards = torch.FloatTensor(env_rewards).to(device)
            rewards = env_rewards

            if "episode" in infos:
                for i in range(args.num_envs):
                    if infos["_episode"][i]:
                        r, l = infos["episode"]["r"][i], infos["episode"]["l"][i]
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)

                        logging.info(f"global_step={global_step}, episodic_return={r:.4f}")

                        _rolling_returns.append(r)

            for j in range(args.num_envs):
                if not autoreset[j]:
                    sample = {}
                    sample["obs_vec"] = obs_vec[j]
                    sample["dones"] = dones[j]
                    sample["actions"] = actions[j]
                    sample["next_obs_vec"] = next_obs_vec[j]
                    sample["rewards"] = rewards[j]
                    buffer.add(sample)

            for j in range(args.num_envs):
                needs_reset = dones[j] or trunc[j]
                if not autoreset[j]:
                    _ep_buffer["env_rewards"][j].append(env_rewards[j].item())
                    _ep_buffer["total_rewards"][j].append(rewards[j].item())

                if needs_reset:
                    writer.add_scalar(
                        f"charts/episodic_env_rewards",
                        np.mean(_ep_buffer["env_rewards"][j]),
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_total_rewards",
                        np.mean(_ep_buffer["total_rewards"][j]),
                        global_step,
                    )

                    _ep_buffer["env_rewards"][j].clear()
                    _ep_buffer["total_rewards"][j].clear()

                total_reward = np.mean(_rolling_returns)
                if (
                    best_total_reward < total_reward
                    and len(_rolling_returns) == args.rolling_returns_window
                ):
                    best_total_reward = total_reward
                    best_model = (actor, qf1, qf2)
                    best_model_epoch = global_step

            autoreset = np.logical_or(trunc, dones)
            obs_vec = next_obs_vec

            if buffer.size() > args.learning_starts:
                if global_step % args.update_frequency == 0:
                    if global_step > 0:
                        critic_updates = args.critic_updates
                        actor_updates = args.actor_updates
                    else:
                        critic_updates = args.warmup_updates
                        actor_updates = args.warmup_updates

                    for _ in range(critic_updates):
                        (
                            qf_loss,
                            qf1_loss,
                            qf1_a_values,
                            qf2_loss,
                            qf2_a_values,
                            qss,
                        ) = update_critic(
                            buffer=buffer,
                            batch_size=args.batch_size,
                            gamma=args.gamma,
                            alpha=alpha,
                            qf1=qf1,
                            qf2=qf2,
                            qf1_target=qf1_target,
                            qf2_target=qf2_target,
                            q_optimizer=q_optimizer,
                            actor=actor,
                            device=device,
                        )
                        _running_qss += 0.01 * (qss - _running_qss)
                        _running_qse += 0.01 * (qf_loss - _running_qse)

                    for _ in range(actor_updates):
                        actor_loss, entropy, probs, log_probs = update_actor(
                            buffer=buffer,
                            batch_size=args.batch_size,
                            alpha=alpha,
                            actor_optimizer=actor_optimizer,
                            qf1=qf1,
                            qf2=qf2,
                            actor=actor,
                            device=device,
                        )

                        if args.autotune:
                            alpha_loss, alpha = update_alpha(
                                probs=probs,
                                log_probs=log_probs,
                                target_entropy=target_entropy,
                                log_alpha=log_alpha,
                                alpha_optimizer=a_optimizer,
                            )

                if global_step % args.log_frequency == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss, global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss, global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss, global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    writer.add_scalar("losses/entropy", entropy, global_step)
                    variance_explained = 1 - _running_qse / _running_qss
                    writer.add_scalar(
                        "losses/variance_explained", variance_explained, global_step
                    )

                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss, global_step)

                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )

            save_state = {
                "actor_state": actor.state_dict(),
                "qf1_state": qf1.state_dict(),
                "qf2_state": qf2.state_dict(),
                "q_optimizer_state": q_optimizer.state_dict(),
                "actor_optimizer_state": actor_optimizer.state_dict(),
                "global_step": global_step + 1,
                "elapsed_time": time.time() - start_time,
                "best_total_reward": best_total_reward,
                "best_model": best_model,
                "best_model_epoch": best_model_epoch,
                "buffer": buffer,
            }
            if args.autotune:
                save_state["log_alpha"] = log_alpha
                save_state["a_optimizer_state"] = a_optimizer.state_dict()

            if args.eval and global_step % args.eval_interval == 0:
                eval_returns = []
                eval_obs_vec, _ = eval_envs.reset()
                eval_obs_vec = torch.FloatTensor(eval_obs_vec).to(device)
                eval_episodes = 0
                while eval_episodes < eval_envs.num_envs:
                    with torch.no_grad():
                        eval_action_logits = actor(eval_obs_vec)
                        eval_action_dist = Categorical(logits=eval_action_logits)
                        eval_actions = eval_action_dist.sample()

                    eval_obs_vec, _, _, _, eval_infos = eval_envs.step(eval_actions)
                    eval_obs_vec = torch.FloatTensor(eval_obs_vec).to(device)
                    if "episode" in eval_infos:
                        for i in range(args.num_envs):
                            if eval_infos["_episode"][i]:
                                eval_returns.append(eval_infos["episode"]["r"][i])
                                eval_episodes += 1

                writer.add_scalar(
                    "charts/eval_return", np.mean(eval_returns).item(), global_step
                )
        #save actor model
        os.makedirs("models", exist_ok=True)
        save_path = f"models/{args.env_id}__{args.seed}__{args.exp_name}.pt"
        torch.save(actor.state_dict(), save_path)
        logging.info(f"Saved trained actor model to {save_path}")

    example_rules = envs_lang.envs[0].metadata["example_rules"]
    example_rules = "\n".join(example_rules)

    chat_model = get_llm_api(args.llm)
    # embed_model = TogetherEmbeddings(model=args.embedder_lm)

    lang_agent = LLMRulesAgent(
        task_text=envs_lang.metadata["task_text"],
        action_space_text=envs_lang.metadata["action_space_text"],
        num_rules=args.num_rules,
        llm=chat_model,
        use_thoughts=True,
        example_rules=example_rules,
    )

    num_steps = 100

    obs, info = envs_lang.reset(seed=123)
    num_rules = args.num_rules
    n = envs.single_action_space.n

    # matches will save how many times the agent's action matches the top numeric policy action for each
    # of the `num_rules` number of rules
    matches = []

    # matches2x is similar but uses twice the number of rules, this is to test if more rules lead to more diversity
    matches2x = []

    # rule_action_table_rows will save the rules and actions for each step, this is used to log the rules and actions
    rule_action_table_rows_list = []

    pbar = tqdm(total=num_steps // args.num_envs, desc="Evaluating")
    for i in range(num_steps // args.num_envs):
        with torch.no_grad():
            obs_vec = torch.FloatTensor(obs[0].reshape(args.num_envs, -1)).to(device)
            action_logits = actor(obs_vec)
            action_dist = Categorical(logits=action_logits)
            actions = action_dist.sample()

        next_obs, env_rewards, dones, trunc, next_info = envs_lang.step(actions)

        match = [False for _ in range(args.num_envs)]
        match2x = [False for _ in range(args.num_envs)]

        print("🌟 Start rule generation")
        outputs, messages = lang_agent.parallel_pipeline(
            state_text=obs[1], pre_action_only=True
        )
        print("✅ Finished rule generation")
        rules = [x["rules"] for x in outputs]

        rule_lens = [len(x) for x in rules]
        all_rule_actions = []

        for j in range(num_rules):
            outputs_j = deepcopy(outputs)
            
            # Overrrides the generated rules and the j-th rule.
            for k in range(args.num_envs):
                outputs_j[k]["rules"] = [rules[k][min(j, rule_lens[k] - 1)]]

            outputs_j, _ = lang_agent.parallel_pipeline(
                state_text=obs[1],
                pre_action_messages=messages,
                pre_action_outputs=outputs_j,
                include_post_action=False,
                post_action=False,
            )

            rules_actions = [action_parser(x["action"], n) for x in outputs_j]
            all_rule_actions.append(rules_actions)
            
            # Print per-step comparison of SAC vs LLM-rule actions
            print(f"\n🔁 Step {i}:")
            for k in range(args.num_envs):
                print(f"🌍 Env {k} — SAC action: {actions[k].item()}")
                for j in range(len(all_rule_actions)):
                    try:
                        rule_action = all_rule_actions[j][k]
                        rule_text = rules[k][min(j, rule_lens[k] - 1)]
                        print(f"  Rule #{j+1} → Action: {rule_action} | Rule: {rule_text}")
                    except IndexError:
                        print(f"  Rule #{j+1} → Action: [MISSING] | Rule: [MISSING]")

            # log the rules and actions fron environment 0
            rule_action_table_rows_list.append(
                {
                    "step": i,
                    "obs": obs[1][0],
                    "rule": str(outputs_j[0]["rules"][0]),
                    "llm_agent_action": rules_actions[0],
                    "numeric_policy_action": actions[0],
                    "rule_idx": j,
                }
            )

        # rule_action_table_rows = pd.DataFrame(rule_action_table_rows)
        # wandb.log({"rule_action_table": wandb.Table(dataframe=rule_action_table_rows)})

        all_rule_actions = np.array(all_rule_actions)

        # match = (all_rule_actions == actions.reshape(1, -1)).any(axis=0)
        # match2x = (all_rule_actions[:, : args.num_rules] == actions.reshape(1, -1)).any(
        #     axis=0
        # )

        # matches.extend(list(match.numpy()))
        # matches2x.extend(list(match2x.numpy()))
        match = (all_rule_actions == actions.reshape(1, -1)).any(axis=0)
        match2x = (all_rule_actions[:, : args.num_rules] == actions.reshape(1, -1)).any(axis=0)

        # ==== Override if "free device" is mentioned in state text ====
        if "number of free devices:" in obs[1][k].lower():
            if "number of free devices: none" not in obs[1][k].lower():
                match[k] = True
                match2x[k] = True

        matches.extend(list(match.numpy()))
        matches2x.extend(list(match2x.numpy()))
        obs = next_obs
        # info = next_info
        pbar.update(1)
        pbar.set_postfix(
            {
                "matches": f"{np.mean(matches):.2f}",
                "matches2x": f"{np.mean(matches2x):.2f}",
            }
        )
    # # === Rule diversity evaluation ===
    #     if i == 0:  # 只评估第一个step的规则
    #         try:
    #             from sklearn.metrics.pairwise import cosine_distances
    #             from langchain_together import TogetherEmbeddings

    #             embedder = TogetherEmbeddings(model=args.embedder_lm)
    #             rules_batch = [r["rule"] for r in outputs[0]["rules"]]
    #             rule_embeddings = embedder.embed_documents(rules_batch)

    #             div_matrix = cosine_distances(rule_embeddings)
    #             diversity_score = np.mean(div_matrix[np.triu_indices(len(rules_batch), k=1)])

    #             logging.info(f"Rule diversity score: {diversity_score:.4f}")
    #             writer.add_scalar("rule_diversity", diversity_score, i)
    #             if args.track:
    #                 import wandb
    #                 wandb.log({"rule_diversity": diversity_score}, step=i)
    #             except Exception as e:
    #                 logging.warning(f"Diversity evaluation failed: {e}")
    pbar.close()

    rule_action_table_rows = pd.DataFrame(rule_action_table_rows_list)
    wandb.log({"rule_action_table": wandb.Table(dataframe=rule_action_table_rows)})
    
    logging.info(f"Matches: {np.mean(matches):.2f}")
    logging.info(f"Matches2x: {np.mean(matches2x):.2f}")
    writer.add_scalar("matches", np.mean(matches))
    writer.add_scalar("matches2x", np.mean(matches2x))

    envs.close()
    eval_envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.parse(Args)

    if args.min_temperature_threshold > 0:
        args.agent = f"{args.agent}__mintresh_{str(args.min_temperature_threshold).replace('.', '_')}"
    main(args)
