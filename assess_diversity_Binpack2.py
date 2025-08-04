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
    cuda: bool = True
    """If toggled, CUDA will be enabled by default."""
    track: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases."""
    wandb_project_name: str = "rulebots"
    """The Weights and Biases project name."""
    wandb_entity: Optional[str] = None
    """The entity (team) of the Weights and Biases project."""
    log_frequency: int = 128
    """The logging frequency of the algorithm."""
    num_diversity_steps = 200
    """The number of steps of the diversity experiment"""

    # Environment
    env_id: str = "BinPacking"
    """The ID of the environment."""
    num_envs: int = 1
    """The number of parallel game environments."""
    max_episode_steps: Optional[int] = 32
    """The maximum number of steps per episode."""
    min_temperature_threshold: float = 0.0
    """Only used for the heat alert environment."""

    # Algorithm
    total_timesteps: int = 200_000
    """Total timesteps of the experiments."""
    gamma: float = 0.95
    """The discount factor gamma."""
    tau: float = 0.25
    """Target smoothing coefficient."""
    batch_size: int = 64
    """The batch size of samples from the replay memory."""
    learning_starts: int = 512
    """Timestep to start learning."""
    policy_lr: float = 1e-4
    """The learning rate of the policy network optimizer."""
    q_lr: float = 1e-4
    """The learning rate of the Q-network optimizer."""
    update_frequency: float | int = 32
    """The frequency of training updates."""
    warmup_updates: int = 1
    """The number of warmup updates to the value function on the first iteration."""
    actor_updates: int = 1
    """The number of updates to the actor per update cycle."""
    critic_updates: int = 1
    """The number of updates to the critic per update cycle."""
    target_network_frequency: int = 256
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
    eval_interval: int = 5_000
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
    hidden_dim: int = 64
    """The hidden dimension of the networks."""

    gpu_memory_utilization: float = 0.9
    """GPU memory to reserve per process"""

    # Buffer collection mode
    load_buffer: bool = False
    """If toggled, the agent will load the buffer from the pickle file if it exists."""
    buffer_size: int = 4096
    """The replay memory buffer size."""

    agent: Optional[str] = "sac_numeric"
    """The agent to use."""
    thoughts: bool = False
    """If toggled, the agent will use thoughts."""

    # Torch compile
    compile_torch: bool = False
    """If toggled, the models will be compiled with Torch."""
    
    # Model persistence
    overwrite_model: bool = False
    """If toggled, the agent will overwrite existing saved models and retrain from scratch."""


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


def get_default_model_path(args: Args) -> str:
    """
    Generate the default model path based on experiment parameters.
    """
    return f"models/{args.env_id}__{args.seed}__{args.exp_name}.pt"


def model_exists(model_path: str) -> bool:
    """
    Check if a model file exists.
    """
    return os.path.exists(model_path)


def apply_action_mask(action_logits, action_mask, mask_value=-1e8):
    """
    Apply action mask to logits by setting invalid actions to a very low value.
    
    Args:
        action_logits: Raw action logits from the actor
        action_mask: Binary mask where 1 indicates valid actions, 0 invalid
        mask_value: Value to set for invalid actions
    
    Returns:
        Masked action logits
    """
    if action_mask is None:
        return action_logits
    
    # Convert action_mask to same device and dtype as logits
    if isinstance(action_mask, np.ndarray):
        action_mask = torch.from_numpy(action_mask).to(action_logits.device)
    
    # Set invalid actions to very low value
    masked_logits = action_logits.clone()
    masked_logits[action_mask == 0] = mask_value
    
    return masked_logits


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
    Modified to handle action masking for BinPacking environment.

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
        # Handle action masking for next state
        if "action_mask" in data and "next_action_mask" in data:
            next_action_mask = data["next_action_mask"]
            raw_logits = actor(next_obs_vec)
            masked_logits = apply_action_mask(raw_logits, next_action_mask)
            dist = Categorical(logits=masked_logits)
        else:
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
    Modified to handle action masking for BinPacking environment.

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

    # Handle action masking
    if "action_mask" in data:
        action_mask = data["action_mask"]
        raw_logits = actor(obs_vec)
        masked_logits = apply_action_mask(raw_logits, action_mask)
        dist = Categorical(logits=masked_logits)
    else:
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


def extract_observation_and_mask(obs):
    """
    Extract observation and action mask from environment observation.
    For BinPacking with action masking, obs is a dict with 'real_obs' and 'action_mask'.
    For other environments, obs is just the observation vector.
    """
    if isinstance(obs, dict) and 'real_obs' in obs:
        return obs['real_obs'], obs.get('action_mask', None)
    else:
        return obs, None


def main(args: Args):
    run_name = f"{args.env_id}__{args.seed}__{args.exp_name}__{args.llm}__{int(time.time())}"
    # normalize run name
    run_name = run_name.replace("/", "_").replace(" ", "_").replace(".", "_")
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
    logging.info(f"Using device: {device}")

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

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    # For BinPacking environment, we need to handle the observation shape differently
    # depending on whether action masking is used
    sample_obs, _ = envs.reset()
    sample_obs_vec, sample_action_mask = extract_observation_and_mask(sample_obs[0])
    obs_shape = sample_obs_vec.shape[-1] if hasattr(sample_obs_vec, 'shape') else len(sample_obs_vec)

    actor = nn.Sequential(
        nn.LayerNorm(obs_shape),
        layer_init(nn.Linear(obs_shape, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), std=1.0),
    )

    qf1 = nn.Sequential(
        nn.LayerNorm(obs_shape),
        layer_init(nn.Linear(obs_shape, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), 0.01),
    )

    qf2 = nn.Sequential(
        nn.LayerNorm(obs_shape),
        layer_init(nn.Linear(obs_shape, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), 0.01),
    )

    # Move networks to device (GPU if available and cuda=True)
    actor = actor.to(device)
    qf1 = qf1.to(device)
    qf2 = qf2.to(device)

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
        ).to(device)
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
    
    # Move target networks to device
    qf1_target = qf1_target.to(device)
    qf2_target = qf2_target.to(device)

    if args.compile_torch:
        actor = torch.compile(actor)
        qf1 = torch.compile(qf1)
        qf2 = torch.compile(qf2)
        qf1_target = torch.compile(qf1_target)
        qf2_target = torch.compile(qf2_target)

    obs, _ = envs.reset()
    obs_vec, action_mask = extract_observation_and_mask(obs)
    obs_vec = torch.FloatTensor(obs_vec).to(device)
    autoreset = np.zeros(args.num_envs, dtype=bool)

    _ep_buffer = defaultdict(lambda: [[] for _ in range(args.num_envs)])
    _rolling_returns = deque(maxlen=args.rolling_returns_window)

    _running_qss = 0.0
    _running_qse = 0.0

    # Get default model path
    default_model_path = get_default_model_path(args)

    # Check if we should load an existing model or train from scratch
    should_train = True
    if not args.overwrite_model and model_exists(default_model_path):
        try:
            actor.load_state_dict(torch.load(default_model_path, map_location=device))
            actor.eval()
            logging.info(f"Loaded existing actor model from {default_model_path}")
            should_train = False
        except Exception as e:
            logging.warning(f"Failed to load existing model: {e}. Will train from scratch.")
            should_train = True

    # === 👇 load actor if eval_only is on ===
    if args.eval_only:
        if args.load_model_path is not None:
            model_path = args.load_model_path
        else:
            model_path = default_model_path

        if not model_exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Either train the model first or provide a valid --load_model_path.")

        actor.load_state_dict(torch.load(model_path, map_location=device))
        actor.eval()
        logging.info(f"Loaded actor model from {model_path}")
        should_train = False

    if should_train:
        for global_step in tqdm(range(starting_step, args.total_timesteps)):
            with torch.no_grad():
                # Apply action masking if available
                if action_mask is not None:
                    raw_logits = actor(obs_vec)
                    masked_logits = apply_action_mask(raw_logits, action_mask)
                    action_dist = Categorical(logits=masked_logits)
                else:
                    action_logits = actor(obs_vec)
                    action_dist = Categorical(logits=action_logits)
                actions = action_dist.sample()

            next_obs, env_rewards, dones, trunc, infos = envs.step(actions)
            
            # Extract observation and mask for next state
            next_obs_vec, next_action_mask = extract_observation_and_mask(next_obs)
            
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
                    
                    # Add action masks to buffer if available
                    if action_mask is not None:
                        sample["action_mask"] = torch.FloatTensor(action_mask[j]).to(device)
                    if next_action_mask is not None:
                        sample["next_action_mask"] = torch.FloatTensor(next_action_mask[j]).to(device)
                        
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

            autoreset = np.logical_or(trunc, dones.cpu().numpy())
            obs_vec = next_obs_vec
            action_mask = next_action_mask

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
                eval_obs, _ = eval_envs.reset()
                eval_obs_vec, eval_action_mask = extract_observation_and_mask(eval_obs)
                eval_obs_vec = torch.FloatTensor(eval_obs_vec).to(device)
                eval_episodes = 0
                while eval_episodes < eval_envs.num_envs:
                    with torch.no_grad():
                        if eval_action_mask is not None:
                            eval_raw_logits = actor(eval_obs_vec)
                            eval_masked_logits = apply_action_mask(eval_raw_logits, eval_action_mask)
                            eval_action_dist = Categorical(logits=eval_masked_logits)
                        else:
                            eval_action_logits = actor(eval_obs_vec)
                            eval_action_dist = Categorical(logits=eval_action_logits)
                        eval_actions = eval_action_dist.sample()

                    eval_next_obs, _, _, _, eval_infos = eval_envs.step(eval_actions)
                    eval_obs_vec, eval_action_mask = extract_observation_and_mask(eval_next_obs)
                    eval_obs_vec = torch.FloatTensor(eval_obs_vec).to(device)
                    if "episode" in eval_infos:
                        for i in range(args.num_envs):
                            if eval_infos["_episode"][i]:
                                eval_returns.append(eval_infos["episode"]["r"][i])
                                eval_episodes += 1

                writer.add_scalar(
                    "charts/eval_return", np.mean(eval_returns).item(), global_step
                )

        # Save actor model
        os.makedirs("models", exist_ok=True)
        torch.save(actor.state_dict(), default_model_path)
        logging.info(f"Saved trained actor model to {default_model_path}")

    example_rules = envs_lang.envs[0].metadata["example_rules"]
    example_rules = "\n".join(example_rules)

    chat_model = get_llm_api(args.llm, gpu_memory_utilization=args.gpu_memory_utilization)
    # embed_model = TogetherEmbeddings(model=args.embedder_lm)

    lang_agent = LLMRulesAgent(
        task_text=envs_lang.metadata["task_text"],
        action_space_text=envs_lang.metadata["action_space_text"],
        num_rules=args.num_rules,
        llm=chat_model,
        use_thoughts=False,
        example_rules=example_rules,
    )


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

    pbar = tqdm(total=args.num_diversity_steps // args.num_envs, desc="Evaluating")
    for i in range(args.num_diversity_steps // args.num_envs):
        # Extract observation vector and action mask for the numeric environment
        obs_vec_lang, action_mask_lang = extract_observation_and_mask(obs[0])
        
        with torch.no_grad():
            obs_vec = torch.FloatTensor(obs_vec_lang.reshape(args.num_envs, -1)).to(device)
            
            # Apply action masking if available
            if action_mask_lang is not None:
                raw_logits = actor(obs_vec)
                masked_logits = apply_action_mask(raw_logits, action_mask_lang)
                action_dist = Categorical(logits=masked_logits)
            else:
                action_logits = actor(obs_vec)
                action_dist = Categorical(logits=action_logits)
            actions = action_dist.sample()

        next_obs, env_rewards, dones, trunc, next_info = envs_lang.step(actions)

        # match = [False for _ in range(args.num_envs)]
        # match2x = [False for _ in range(args.num_envs)]

        print("🌟 Start rule generation")
        outputs, messages = lang_agent.parallel_pipeline(
            state_text=obs[1], pre_action_only=True
        )
        print("✅ Finished rule generation")
        rules = [x["rules"] for x in outputs]

        rule_lens = [len(x) for x in rules]
        all_rule_actions = []

        for j in tqdm(range(args.num_envs), desc="Generating rules", leave=False):
            # rules_j = rules[j]
            # outputs_j = deepcopy(outputs)

            # # Overrrides the generated rules and the j-th rule.
            # for k in range(args.num_envs):
            #     outputs_j[k]["rules"] = [rules[k][min(j, rule_lens[k] - 1)]]

            # outputs_j, _ = lang_agent.parallel_pipeline(
            #     state_text=obs[1],
            #     pre_action_messages=messages,
            #     pre_action_outputs=outputs_j,
            #     include_post_action=False,
            #     post_action=False,
            # )

            rules_actions = []
            for m, x in enumerate(rules[j]):
                # use re to find the first integer after the word "action", ther ecould be other characters in between
                try:
                    import json
                    raw = json.loads(x)
                    raw = raw["action"] if "action" in raw else re.findall(r"\d+", str(raw["actions"]))
                except:
                    # Search for 'action' or 'actions' with quotes and a number after the colon
                    raw = re.search(r'["\']actions?["\']\s*:\s*(\d+)', x, re.IGNORECASE)
                    if raw:
                        raw = raw.group(1)
                    else:
                        raw = re.findall(r"\d+", str(x))

                if isinstance(raw, list):
                    rules_actions.append(raw)
                else:
                    # fallback: 解析成 list[int]
                    extracted = [int(i) for i in re.findall(r"\d+", str(raw))]
                    rules_actions.append(extracted)

                # Check if there was a free device to check for ties (BinPacking specific logic)
                free_device = False
                if (
                    "Number of free devices:" in obs[1][j]
                    and "Number of free devices: none" not in obs[1][j]
                ):
                    free_device = True

                # log the rules and actions from environment j
                rule_action_table_rows_list.append(
                    {
                        "environment": j,
                        "step": i,
                        "obs": obs[1][j],
                        "rule": x,
                        "llm_agent_action": [int(u) for u in rules_actions[-1]],  
                        "numeric_policy_action": int(actions[0]),
                        "rule_idx": m,
                        "free_device": free_device,
                    }
                )

            all_rule_actions.append(rules_actions)

        obs = next_obs
        # info = next_info
        pbar.update(1)
        pbar.set_postfix(
            {
                "matches": f"{np.mean(matches):.2f}",
                "matches2x": f"{np.mean(matches2x):.2f}",
            }
        )

    pbar.close()

    rule_action_table_rows = pd.DataFrame(rule_action_table_rows_list)
    wandb.log({"rule_action_table": wandb.Table(dataframe=rule_action_table_rows)})
    rule_action_table_rows.to_parquet(f"logs/diversity/{run_name}.parquet")

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
