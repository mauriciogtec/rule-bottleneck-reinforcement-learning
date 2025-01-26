# code inspired by cleanrl's sac atari implementation
import logging
import os
import pickle
import random
import shutil
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import gymnasium as gym
import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchsummary
import tyro
from langchain_together import TogetherEmbeddings
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import buffers
import envs as E  # registers the gym environments during import
from agents import RulesSelectorActorCritic, PureLanguageAgents
from layers import CrossAttentionNetwork
from llm_apis import ValidLLMs, get_llm_api

# configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rulebots"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    log_frequency: int = 1
    """the logging frequency of the algorithm"""
    log_examples_interval: int = 20
    """the logging frequency of the examples"""
    resume: bool = False
    """if toggled, tries to resume training from the latest checkpoint"""
    ckpt_interval: int = 200
    """the saving interval of the model"""
    overwrite_ckpt: bool = False
    """if toggled and resuming is on, it will start fresh in resume mode, otherwise ignored"""

    # Environment
    env_id: str = "Uganda"
    """the id of the environment"""
    num_envs: int = 4
    """the number of parallel game environments"""
    parallel_pipeline: bool = True
    """if toggled, the pipeline will be parallelized"""
    max_episode_steps: Optional[int] = 16
    """the maximum number of steps per episode"""

    # Algorithm
    total_timesteps: int = 1280
    """total timesteps of the experiments"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 16
    """the batch size of sample from the reply memory"""
    learning_starts: int = 256
    """timestep to start learning"""
    policy_lr: float = 1e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: float | int = 1
    """the frequency of training updates"""
    warmup_updates: int = 1
    """the number of warmup updates to the value function on the first iteration."""
    actor_updates: int = 16
    """the number of updates to the actor per update cycle"""
    critic_updates: int = 4
    """the number of updates to the critic per update cycle"""
    target_network_frequency: int = 64
    """the frequency of updates for the target networks"""
    alpha: float = 0.01
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""
    dropout: float = 0.05
    """the dropout rate"""

    # Eval
    eval: bool = True
    """if toggled, the agent will be evaluated"""
    num_eval_episodes: int = 4
    """the number of episodes to evaluate the agent"""
    eval_interval: int = 64
    """the evaluation interval"""
    eval_deterministic: bool = True
    """if toggled, the evaluation will be deterministic"""
    rolling_returns_window: int = 16
    """the rolling rewards window"""
    proj_type: Literal["linear", "random"] = "linear"
    """if toggled, the agent will use random projection"""

    # LLM
    num_rules: int = 5
    """The number of rules for rule-based LLM-only agent"""
    llm: ValidLLMs = "gpt-4o-mini-huit"
    """the language model to use"""
    embedder_lm: str = "togethercomputer/m2-bert-80M-8k-retrieval"
    """the language model to use for embeddings"""
    embed_dim: int = 768
    """the dimension of the embeddings"""
    hidden_dim: int = 16
    """the hidden dimension of the networks"""
    rule_reward_coef: float = 1.0
    """the reward coefficient for the rules"""
    in_context_learning: bool = True
    """if toggled, the agent will learn in context"""

    # Buffer collection mode
    buffer_collection_steps: int = 64
    """the number of steps to collect data to the buffer"""
    load_buffer: bool = False
    """if toggled, the agent will load the buffer from the pickle file if it exists"""
    buffer_size: int = 4096
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway

    agent: Optional[str] = None  # to be set by the agent
    """the agent to use"""
    thoughts: bool = True
    """if toggled, the agent will use thoughts"""

    # Torch compile
    compile_torch: bool = False  # needs fix


def make_env(env_id, seed, max_episode_steps=None):
    def scale_reward(r):
        return r / max_episode_steps

    def thunk():
        env = gym.make(env_id)
        if env_id == "HeatAlerts":
            env = gym.wrappers.TransformReward(env, func=scale_reward)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env

    return thunk


def save_checkpoint(state, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location=device)
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
        return None


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
    device,
    lang_agent,
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
        device: Device for tensor computations.
        lang_agent: Language agent to calculate policy from embeddings.

    Returns:
        qf_loss: Combined loss for both Q networks.
        qf1_loss: Loss for Q-network 1.
        qf2_loss: Loss for Q-network 2.
    """
    qf1.train()
    qf2.train()
    data = buffer.sample(batch_size)
    next_obs_vec = (
        data["next_obs_vec"].unsqueeze(1)
        if data["next_obs_vec"].dim() == 2
        else data["next_obs_vec"]
    )
    rules_emb = data["rules_emb"]
    rewards = data["rewards"]
    dones = data["dones"]

    with torch.no_grad():
        dist = lang_agent.get_policy_from_embeddings(next_obs_vec, rules_emb)
        next_action_probs = F.softmax(dist.logits, dim=-1)
        next_state_log_pi = F.log_softmax(dist.logits, dim=-1)

        qf1_next_tgt = qf1_target(rules_emb, next_obs_vec)
        qf2_next_tgt = qf2_target(rules_emb, next_obs_vec)

        # Handle nested tensors
        if qf1_next_tgt.is_nested:
            qf1_next_tgt = torch.nested.to_padded_tensor(qf1_next_tgt, 0.0)
            qf2_next_tgt = torch.nested.to_padded_tensor(qf2_next_tgt, 0.0)

        min_qf_next_target = (
            next_action_probs
            * (torch.min(qf1_next_tgt, qf2_next_tgt) - alpha * next_state_log_pi)
        ).sum(dim=1)
        next_q_value = rewards + (1 - dones) * gamma * min_qf_next_target

    obs_vec = (
        data["obs_vec"].unsqueeze(1) if data["obs_vec"].dim() == 2 else data["obs_vec"]
    )
    sel_idxs = data["sel_idxs"].unsqueeze(-1).to(device)

    qf1_values = qf1(rules_emb, obs_vec)
    qf2_values = qf2(rules_emb, obs_vec)

    if qf1_values.is_nested:
        qf1_values = torch.nested.to_padded_tensor(qf1_values, 0.0)
        qf2_values = torch.nested.to_padded_tensor(qf2_values, 0.0)

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
    lang_agent,
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
        lang_agent: Language agent to calculate policy from embeddings.
        device: Device for tensor computations.

    Returns:
        actor_loss: Loss for the actor network.
    """
    lang_agent.actor.train()
    data = buffer.sample(batch_size)
    obs_vec = (
        data["obs_vec"].unsqueeze(1) if data["obs_vec"].dim() == 2 else data["obs_vec"]
    )
    rules_emb = data["rules_emb"]

    dist = lang_agent.get_policy_from_embeddings(obs_vec, rules_emb)
    log_probs = F.log_softmax(dist.logits, dim=-1)
    probs = dist.probs

    with torch.no_grad():
        qf1_values = qf1(rules_emb, obs_vec)
        qf2_values = qf2(rules_emb, obs_vec)

        if qf1_values.is_nested:
            qf1_values = torch.nested.to_padded_tensor(qf1_values, 0.0)
            qf2_values = torch.nested.to_padded_tensor(qf2_values, 0.0)

        min_qf_values = torch.min(qf1_values, qf2_values)

    actor_loss = (probs * (alpha * log_probs - min_qf_values)).mean()
    entropy = dist.entropy().mean().item()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    lang_agent.actor.eval()

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
    run_id = f"{args.agent}__{args.env_id}__{args.exp_name}__{args.seed}"
    run_name = run_id if args.resume else f"{run_id}__{int(time.time())}"

    ckpt_path = f"checkpoints/{run_name}.state"
    text_logs_path = f"text_logs/{run_name}.jsonl"
    json_logger_mode = "w" if not args.resume else "a"
    os.makedirs(os.path.dirname(text_logs_path), exist_ok=True)
    jsonl_logger = jsonlines.open(text_logs_path, mode=json_logger_mode)

    if args.overwrite_ckpt:
        # delete checkpoint pat
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        # delete tensorboard logs
        dirname = f"runs/{run_name}"
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

    if args.track:
        import wandb
        # replace : with - to avoid wandb bug
        wandb_run_name = run_name.replace(":", "-")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=wandb_run_name,
            id=wandb_run_name,
            monitor_gym=True,
            save_code=False,
            resume='auto',
            settings=wandb.Settings(init_timeout=1200, _service_wait=600),
        )
        examples_table = wandb.Table(columns=["global_step", "run_id", "example"])
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    dev = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    train_env_funs = [
        make_env(args.env_id, args.seed + i, args.max_episode_steps)
        for i in range(args.num_envs)
    ]
    eval_env_funs = [
        make_env(args.env_id, 1000 * args.seed + i, args.max_episode_steps)
        for i in range(args.num_envs)
    ]
    envs = gym.vector.SyncVectorEnv(train_env_funs)
    eval_envs = gym.vector.SyncVectorEnv(eval_env_funs)

    # setup language model
    chat_model = get_llm_api(args.llm)
    embed_model = TogetherEmbeddings(model=args.embedder_lm)

    # problem dimensions
    state_dim = envs.single_observation_space[0].shape[-1]
    rule_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    num_rules = args.num_rules
    # num_actions = envs.single_action_space.n

    actor = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        proj_type=args.proj_type,
    )
    qf1 = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        proj_type=args.proj_type,
    )
    qf2 = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        proj_type=args.proj_type,
    )
    actor.eval()
    qf1.eval()
    qf2.eval()

    logging.info("--- Actor ---")
    torchsummary.summary(actor)

    logging.info("--- Q-function ---")
    torchsummary.summary(qf1)

    # language agent
    example_rules = envs.envs[0].metadata["example_rules"]
    example_rules = "\n".join(example_rules)

    def critic(rules_emb, obs_vec):
        q1 = qf1(rules_emb, obs_vec)
        q2 = qf2(rules_emb, obs_vec)
        return torch.min(q1, q2)

    lang_agent = RulesSelectorActorCritic(
        actor=actor,
        task_text=envs.envs[0].metadata["task_text"],
        action_space_text=envs.envs[0].metadata["action_space_text"],
        num_rules=num_rules,
        llm=chat_model,
        embededder=embed_model,
        max_rule_combinations=1,
        example_rules=example_rules,
        use_thoughts=args.thoughts,
        critic=critic,
        in_context_learning=args.in_context_learning,
    )

    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_params = list(qf1.parameters()) + list(qf2.parameters())
    q_optimizer = optim.Adam(q_params, lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(
            1 / torch.tensor(envs.single_action_space.n)
        )
        log_alpha = torch.scalar_tensor(
            np.log(args.alpha), requires_grad=True, device=dev
        )
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    buffer_file = f"buffers/buffer_{args.llm}.pkl"
    if args.load_buffer and os.path.exists(buffer_file):
        with open(buffer_file, "rb") as f:
            buffer = pickle.load(f)
        needs_save_buffer = False
    else:
        buffer = buffers.SimpleDictReplayBuffer(args.buffer_size, device=dev)
        needs_save_buffer = True

    starting_step = 0
    start_time = time.time()
    best_total_reward = -float("inf")
    best_model = None
    best_model_epoch = -1

    checkpoint = load_checkpoint(ckpt_path, dev)
    if args.resume and checkpoint:
        logging.info(
            f"Resuming training from checkpoint at step {checkpoint['global_step']}."
        )
        actor.load_state_dict(checkpoint["actor_state"])
        qf1.load_state_dict(checkpoint["qf1_state"])
        qf2.load_state_dict(checkpoint["qf2_state"])
        if args.autotune:
            log_alpha = checkpoint["log_alpha"]
            alpha = log_alpha.exp().item()
            a_optimizer.load_state_dict(checkpoint["a_optimizer_state"])
        q_optimizer.load_state_dict(checkpoint["q_optimizer_state"])
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
        starting_step = checkpoint["global_step"]
        start_time = time.time() - checkpoint["elapsed_time"]
        best_total_reward = checkpoint["best_total_reward"]
        best_model = checkpoint["best_model"]
        best_model_epoch = checkpoint["best_model_epoch"]
        buffer = checkpoint["buffer"]
        logging.info(f"Resumed training from checkpoint at step {starting_step}.")

    logging.info(f"Starting buffer size: {buffer.size()}")

    qf1_target = deepcopy(qf1)
    qf2_target = deepcopy(qf2)

    # compile the models with torch > 2
    if args.compile_torch:
        actor = torch.compile(actor)
        qf1 = torch.compile(qf1)
        qf2 = torch.compile(qf2)
        qf1_target = torch.compile(qf1_target)
        qf2_target = torch.compile(qf2_target)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    obs_vec, obs_text = obs

    # convert to torch tensor
    obs_vec = torch.FloatTensor(obs_vec).to(dev)

    # expand state space with the rules, i.e., internal states
    with torch.no_grad():
        outputs, messages = lang_agent.parallel_pipeline(
            obs_text, obs_vec, pre_action_only=True
        )
    rules = [x["rules"] for x in outputs]
    rules_emb = [x["rules_emb"] for x in outputs]
    sel_idxs = [x["sel_idx"] for x in outputs]
    autoreset = np.zeros(args.num_envs, dtype=bool)

    # keep logging buffers for the rewards
    _ep_buffer = defaultdict(lambda: [[] for _ in range(args.num_envs)])
    _rolling_returns = deque(maxlen=args.rolling_returns_window)

    _running_qss = 0.0
    _running_qse = 0.0

    for global_step in tqdm(range(starting_step, args.total_timesteps)):
        with torch.no_grad():
            outputs, messages = lang_agent.parallel_pipeline(
                state_text=obs_text,
                state_vector=obs_vec,
                pre_action_outputs=outputs,
                pre_action_messages=messages,
            )
        actions = [x["action"] for x in outputs]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, env_rewards, dones, trunc, infos = envs.step(actions)
        next_obs_vec, next_obs_text = next_obs
        dones = torch.FloatTensor(dones).to(dev)
        next_obs_vec = torch.FloatTensor(next_obs_vec).to(dev)

        env_rewards = torch.FloatTensor(env_rewards).to(dev)
        sel_rewards = torch.FloatTensor([x["sel_reward"] for x in outputs]).to(dev)
        sel_reward_scores = [x["sel_reward_scores"] for x in outputs]
        rewards = env_rewards + args.rule_reward_coef * sel_rewards
        entropy = [x["entropy"] for x in outputs]
        sel_probs = [x["sel_logprob"].exp() for x in outputs]
        _rolling_returns.extend(list(rewards.cpu().numpy()))

        # Get the next rules
        outputs = deepcopy(outputs)
        messages = deepcopy(messages)
        with torch.no_grad():
            next_outputs, next_messages = lang_agent.parallel_pipeline(
                next_obs_text, next_obs_vec, pre_action_only=True
            )
        next_rules = [x["rules"] for x in next_outputs]
        next_rules_emb = [x["rules_emb"] for x in next_outputs]
        next_sel_idxs = [x["sel_idx"] for x in next_outputs]

        if "episode" in infos:
            for i in range(args.num_envs):
                if infos["_episode"][i]:
                    r, l = infos["episode"]["r"][i], infos["episode"]["l"][i]
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)

                    logging.info(f"global_step={global_step}, episodic_return={r:.4f}")

        for j in range(args.num_envs):
            if not autoreset[j]:
                sample = {}
                sample["obs_vec"] = obs_vec[j]
                sample["obs_text"] = obs_text[j]
                sample["rules"] = rules[j]
                sample["rules_emb"] = rules_emb[j]
                sample["dones"] = dones[j]
                sample["sel_idxs"] = sel_idxs[j]
                sample["actions"] = actions[j]
                sample["next_obs_vec"] = next_obs_vec[j]
                sample["next_obs_text"] = next_obs_text[j]
                sample["next_rules_emb"] = next_rules_emb[j]
                sample["next_rules"] = next_rules[j]
                sample["rewards"] = rewards[j]
                sample["sel_rewards"] = sel_rewards[j]
                sample["env_rewards"] = env_rewards[j]
                buffer.add(sample)

        # accumulate and log the rewards
        for j in range(args.num_envs):
            needs_reset = dones[j] or trunc[j]
            if not needs_reset:
                _ep_buffer["env_rewards"][j].append(env_rewards[j].item())
                _ep_buffer["sel_rewards_scores"][j].append(sel_reward_scores[j])
                _ep_buffer["sel_rewards_total"][j].append(sel_rewards[j].item())
                _ep_buffer["total_rewards"][j].append(rewards[j].item())
                _ep_buffer["entropy"][j].append(entropy[j].item())
                _ep_buffer["sel_probs"][j].append(sel_probs[j].item())
            else:
                # log the rewards
                writer.add_scalar(
                    f"charts/episodic_env_rewards",
                    np.mean(_ep_buffer["env_rewards"][j]),
                    global_step,
                )
                m = np.mean(_ep_buffer["sel_rewards_scores"][j], axis=0)
                for i, x in enumerate(m):
                    writer.add_scalar(f"charts/sel_reward_scores/q{i}", x, global_step)
                m = np.mean(_ep_buffer["sel_rewards_total"][j])
                writer.add_scalar("charts/episodic_sel_rewards", m, global_step)
                writer.add_scalar(
                    "charts/episodic_entropy",
                    np.mean(_ep_buffer["entropy"][j]),
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_sel_probs",
                    np.mean(_ep_buffer["sel_probs"][j]),
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_total_rewards",
                    np.mean(_ep_buffer["total_rewards"][j]),
                    global_step,
                )

                # flush
                _ep_buffer["env_rewards"][j].clear()
                _ep_buffer["sel_rewards_scores"][j].clear()
                _ep_buffer["sel_rewards_total"][j].clear()
                _ep_buffer["entropy"][j].clear()
                _ep_buffer["sel_probs"][j].clear()
                _ep_buffer["total_rewards"][j].clear()

        # Log
        if global_step % args.log_examples_interval == 0:
            rules_str = "\n".join(outputs[0]["rules"])
            rules_scores = [
                f"Q{d + 1}. {k}: {v}"
                for d, (k, v) in enumerate(outputs[0]["sel_reward_scores_raw"].items())
            ]
            rules_scores_str = "\n".join(rules_scores)
            thoughts = outputs[0].get("thoughts", None)
            example = [
                f"{outputs[0]['initial_prompt']}\n",
                f"### Thoughts\n{thoughts}\n" if thoughts else "",
                f"### Rules\n{rules_str}\n",
                f"### Selected Rule\n{outputs[0]['sel_rule']}\n",
                f"### Selected Rule Probability\n{outputs[0]['sel_logprob'].exp():.2f}\n",
                f"### Selected Rule Reward\n {outputs[0]['sel_reward']}\n",
                f"### Selected Rule Explainability\n{rules_scores_str}\n",
                f"### Environment Action\n{outputs[0]['action']}\n",
                f"### Explanation \n{outputs[0]['explanation']}\n",
            ]
            example = "".join(example)

            conversation = "\n".join(
                [f"\n\n## {x['role']}\n\n{x['content']}" for x in messages[0]]
            )
            writer.add_text("text/examples", example, global_step)
            writer.add_text("llm_prompts/conversation", conversation, global_step)
            if args.track:
                examples_table.add_data(global_step, run_id, example)
                wandb.log({"examples": examples_table})

            # log the conversation and example in jsonl
            jsonl_logger.write(
                {
                    "global_step": global_step,
                    "example": example,
                    "conversation": messages[0],
                }
            )

            # save best model
            total_reward = np.mean(_rolling_returns)
            if (
                best_total_reward < total_reward
                and len(_rolling_returns) == args.rolling_returns_window
            ):
                best_total_reward = total_reward
                best_model = (actor, qf1, qf2)
                best_model_epoch = global_step

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        autoreset = np.logical_or(autoreset, dones)
        obs_vec, obs_text = next_obs_vec, next_obs_text
        rules = next_rules
        rules_emb = next_rules_emb
        sel_idxs = next_sel_idxs
        outputs, messages = next_outputs, next_messages

        # ALGO LOGIC: training.
        if buffer.size() > args.learning_starts:
            if needs_save_buffer:
                os.makedirs(os.path.dirname(buffer_file), exist_ok=True)
                with open(buffer_file, "wb") as f:
                    pickle.dump(buffer, f)
                logging.info(f"Buffer saved to {buffer_file}")
                needs_save_buffer = False
            if global_step % args.update_frequency == 0:
                if global_step > 0:
                    critic_updates = args.critic_updates
                    actor_updates = args.actor_updates
                else:
                    critic_updates = args.warmup_updates
                    actor_updates = args.warmup_updates

                for _ in range(critic_updates):
                    # Update critic
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
                        device=dev,
                        lang_agent=lang_agent,
                    )
                    _running_qss += 0.01 * (qss - _running_qss)
                    _running_qse += 0.01 * (qf_loss - _running_qse)

                for _ in range(actor_updates):
                    # Update actor
                    actor_loss, entropy, probs, log_probs = update_actor(
                        buffer=buffer,
                        batch_size=args.batch_size,
                        alpha=alpha,
                        actor_optimizer=actor_optimizer,
                        qf1=qf1,
                        qf2=qf2,
                        lang_agent=lang_agent,
                        device=dev,
                    )

                    if args.autotune:  # Check if alpha tuning is enabled
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

            # update the target networks
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
        save_checkpoint(save_state, ckpt_path)

        if global_step % args.ckpt_interval == 0:
            save_checkpoint(
                save_state, ckpt_path.replace(".state", f"__{global_step}.state")
            )

        # Evaluation loop
        lang_agent.deterministic = True

        if args.eval and global_step % args.eval_interval == 0:
            eval_returns = []
            eval_obs, _ = eval_envs.reset()
            eval_obs_vec, eval_obs_text = eval_obs
            eval_obs_vec = torch.FloatTensor(eval_obs_vec).to(dev)
            eval_episodes = 0
            while eval_episodes < eval_envs.num_envs:
                with torch.no_grad():
                    eval_outputs, _ = lang_agent.parallel_pipeline(
                        eval_obs_text,
                        eval_obs_vec,
                    )
                eval_actions = [x["action"] for x in eval_outputs]
                eval_obs_vec, _, _, _, eval_infos = eval_envs.step(eval_actions)
                eval_obs_vec, eval_next_obs_vec = eval_obs_vec
                eval_obs_vec = torch.FloatTensor(eval_obs_vec).to(dev)
                if "episode" in eval_infos:
                    for i in range(args.num_envs):
                        if eval_infos["_episode"][i]:
                            eval_returns.append(eval_infos["episode"]["r"][i])
                            eval_episodes += 1

            # log
            writer.add_scalar(
                "charts/eval_return", np.mean(eval_returns).item(), global_step
            )

        lang_agent.deterministic = False

    envs.close()
    eval_envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.parse(Args)

    args.agent = "rbrl"
    if not args.thoughts:
        args.agent += "-no-thoughts"
    if not args.in_context_learning:
        args.agent += "-no-in-context"
    
    args.agent += "--" + args.llm

    main(args)
