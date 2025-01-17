# code inspired by cleanrl's sac atari implementation
import logging
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Optional

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
from agents import RulesSelectorActorCritic, ValidAgents
from layers import AttentionNetwork
from llm_apis import ValidModels, get_llm_api

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
    cuda: bool = True
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
    ckpt_interval: int = 1
    """the saving interval of the model"""
    overwrite_ckpt: bool = False
    """if toggled and resuming is on, it will start fresh in resume mode, otherwise ignored"""

    # Environment
    env_id: str = "Uganda"
    """the id of the environment"""
    num_envs: int = 4
    """the number of parallel game environments"""
    agent: ValidAgents = "llm_rules_agent"
    """the agent to use"""
    parallel_pipeline: bool = True
    """if toggled, the pipeline will be parallelized"""

    # Algorithm
    total_timesteps: int = 1250
    """total timesteps of the experiments"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    learning_starts: int = 64
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: float | int = 1 / 16
    """the frequency of training updates"""
    warmup_updates: int = 64
    """the number of warmup updates to the value function on the first iteration."""
    target_network_frequency: int = 64
    """the frequency of updates for the target networks"""
    alpha: float = 0.01
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""

    # Eval
    num_eval_steps: int = 64
    """the number of steps to run in each eval environment per policy rollout"""
    eval_interval: int = 1
    """the evaluation interval"""
    eval_deterministic: bool = True
    """if toggled, the evaluation will be deterministic"""
    rolling_rewards_window: int = 64
    """the rolling rewards window"""

    # LLM
    num_rules: int = 10
    """The number of rules for rule-based LLM-only agent"""
    llm: ValidModels = "gpt-4o-mini-huit"
    """the language model to use"""
    embedder_lm: str = "togethercomputer/m2-bert-80M-8k-retrieval"
    """the language model to use for embeddings"""
    embed_dim: int = 768
    """the dimension of the embeddings"""
    hidden_dim: int = 16
    """the hidden dimension of the networks"""

    # Buffer collection mode
    buffer_collection_steps: int = 64
    """the number of steps to collect data to the buffer"""
    load_buffer: bool = True
    """if toggled, the agent will load the buffer from the pickle file if it exists"""
    buffer_size: int = 1024
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway

    # Torch compile
    compile_torch: bool = (
        False  # TODO: need to fix the ordering of the save/load/compile to avoid errors
    )


def make_env(env_id, seed, eval=False):
    def thunk():
        if eval:
            env = gym.make(env_id, max_episode_steps=None, T=args.num_eval_steps)
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = E.wrappers.SymlogRewardsWrapper(env)
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


if __name__ == "__main__":
    args = tyro.cli(Args)

    run_id = f"sac_attention_{args.env_id}__{args.exp_name}__{args.llm}__{args.seed}"
    run_name = run_id if args.resume else f"{run_id}_{int(time.time())}"

    ckpt_path = f"checkpoints/best_{run_name}.state"
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    train_env_funs = [
        make_env(args.env_id, args.seed + i) for i in range(args.num_envs)
    ]
    eval_env_funs = [
        make_env(args.env_id, 1000 * args.seed + i, eval=True)
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
    num_actions = envs.single_action_space.n

    actor = AttentionNetwork(
        q_dim=state_dim,
        k_dim=rule_dim,
        hidden_dim=hidden_dim,
        weights_only=True,
    )
    qf1 = AttentionNetwork(
        q_dim=state_dim,
        k_dim=rule_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        weights_only=True,
    )
    qf2 = AttentionNetwork(
        q_dim=state_dim,
        k_dim=rule_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        weights_only=True,
    )

    logging.info("--- Actor ---")
    torchsummary.summary(actor)

    logging.info("--- Q-function ---")
    torchsummary.summary(qf1)

    # language agent
    lang_agent = RulesSelectorActorCritic(
        actor=actor,
        task_text=envs.envs[0].metadata["task_text"],
        action_space_text=envs.envs[0].metadata["action_space_text"],
        num_rules=num_rules,
        llm=chat_model,
        embededder=embed_model,
        max_rule_combinations=1,
    )

    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(
            1 / torch.tensor(envs.single_action_space.n)
        )
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
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
        buffer = buffers.SimpleDictReplayBuffer(args.buffer_size, device=device)
        needs_save_buffer = True

    starting_step = 0
    start_time = time.time()
    best_total_reward = -float("inf")
    best_model = None

    checkpoint = load_checkpoint(ckpt_path, device)
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
    obs_vec = torch.FloatTensor(obs_vec).to(device)

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
    _rolling_rewards = deque(maxlen=args.rolling_rewards_window)
    # _ep_env_rewards = [[] for _ in range(args.num_envs)]
    # _ep_sel_rewards_scores = [[] for _ in range(args.num_envs)]
    # _ep_sel_rewards_total = [[] for _ in range(args.num_envs)]

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
        next_obs, env_rewards, terminations, truncations, infos = envs.step(actions)
        next_obs_vec, next_obs_text = next_obs

        next_obs_vec = torch.FloatTensor(next_obs_vec).to(device)
        terminations = torch.FloatTensor(terminations).to(device)
        env_rewards = torch.FloatTensor(env_rewards).to(device)
        sel_rewards = torch.FloatTensor([x["sel_reward"] for x in outputs]).to(device)
        sel_reward_scores = [x["sel_reward_scores"] for x in outputs]
        rewards = env_rewards + sel_rewards
        entropy = [x["entropy"] for x in outputs]
        sel_probs = [x["sel_logprob"].exp() for x in outputs]
        _rolling_rewards.extend(list(rewards.cpu().numpy()))

        # accumulate and log the rewards
        for j in range(args.num_envs):
            done_now = terminations[j] or truncations[j]
            if not done_now:
                _ep_buffer["env_rewards"][j].append(env_rewards[j].item())
                _ep_buffer["sel_rewards_scores"][j].append(sel_reward_scores[j])
                _ep_buffer["sel_rewards_total"][j].append(sel_rewards[j].item())
                _ep_buffer["entropy"][j].append(entropy[j].item())
                _ep_buffer["sel_probs"][j].append(sel_probs[j].item())
            else:
                # log the rewards
                writer.add_scalar(
                    f"charts/episodic_env_rewards/",
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

                # flush
                _ep_buffer["env_rewards"][j].clear()
                _ep_buffer["sel_rewards_scores"][j].clear()
                _ep_buffer["sel_rewards_total"][j].clear()
                _ep_buffer["entropy"][j].clear()
                _ep_buffer["sel_probs"][j].clear()

        # Log
        if global_step % args.log_examples_interval == 0:
            rules_str = "\n".join(outputs[0]["rules"])
            rules_scores = [
                f"{k}: {v}" for k, v in outputs[0]["sel_reward_scores_raw"].items()
            ]
            rules_scores_str = "\n".join(rules_scores)
            example = (
                f"{outputs[0]['initial_prompt']}\n"
                f"### Thoughts\n{outputs[0]['thoughts']}\n"
                f"### Rules\n{rules_str}\n"
                f"### Selected Rule\n{outputs[0]['sel_rule']}\n"
                f"### Selected Rule Probabßility\n{outputs[0]['sel_logprob'].exp():.2f}\n"
                f"### Selected Rule Reward\n {outputs[0]['sel_reward']}\n"
                f"### Selected Rule Explainability\n{rules_scores_str}\n"
                f"### Environment Action\n{outputs[0]['action']}\n"
                f"### Explanation with thoughts and all rules\n{outputs[0]['explanation']}\n"
                f"### Explanation with only selected rule\n{outputs[0]['explanation_rule_only']}"
            )

            conversation = "\n".join(
                [f"\n\n## {x['role']}\n\n{x['content']}" for x in messages[0]]
            )
            writer.add_text("text/examples", example, global_step)
            writer.add_text("llm_prompts/conversation", conversation, global_step)

            # log the conversation and example in jsonl
            jsonl_logger.write(
                {
                    "global_step": global_step,
                    "example": example,
                    "conversation": messages[0],
                }
            )

        # save best model
        total_reward = np.mean(_rolling_rewards)
        if best_total_reward < total_reward:
            best_total_reward = total_reward
            best_model = (actor, qf1, qf2)

        # Get the next rules
        with torch.no_grad():
            outputs, _ = lang_agent.parallel_pipeline(
                next_obs_text, next_obs_vec, pre_action_only=True
            )
        next_rules = [x["rules"] for x in outputs]
        next_rules_emb = [x["rules_emb"] for x in outputs]
        next_sel_idxs = [x["sel_idx"] for x in outputs]

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
                sample["rules_emb"] = rules_emb[j]
                sample["next_obs_vec"] = next_obs_vec[j]
                sample["next_obs_text"] = next_obs_text[j]
                sample["rules_emb_next"] = next_rules_emb[j]
                sample["actions"] = actions[j]
                sample["sel_idxs"] = sel_idxs[j]
                sample["rewards"] = rewards[j]
                sample["sel_rewards"] = sel_rewards[j]
                sample["env_rewards"] = env_rewards[j]
                sample["dones"] = terminations[j]
                buffer.add(sample)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        autoreset = np.logical_or(terminations, truncations)
        obs_vec, obs_text = next_obs_vec, next_obs_text
        rules = next_rules
        rules_emb = next_rules_emb
        sel_idxs = next_sel_idxs

        # ALGO LOGIC: training.
        if buffer.size() > args.learning_starts:
            if needs_save_buffer:
                os.makedirs(os.path.dirname(buffer_file), exist_ok=True)
                with open(buffer_file, "wb") as f:
                    pickle.dump(buffer, f)
                logging.info(f"Buffer saved to {buffer_file}")
                needs_save_buffer = False
            if global_step % args.update_frequency == 0:
                num_updates = (
                    ceil(1 / args.update_frequency)
                    if global_step > 0
                    else args.warmup_updates
                )
                for _ in range(num_updates):
                    data = buffer.sample(args.batch_size)
                    next_obs_vec = data["next_obs_vec"]
                    next_obs_text = data["next_obs_text"]
                    rules_emb_next = data["rules_emb_next"]

                    if next_obs_vec.dim() == 2:
                        next_obs_vec = next_obs_vec.unsqueeze(1)

                    # CRITIC training
                    with torch.no_grad():
                        dist = lang_agent.get_policy_from_embeddings(
                            next_obs_vec, rules_emb_next
                        )
                        next_pr = F.softmax(dist.logits, dim=-1)
                        entropy = dist.entropy()
                        next_state_log_pi = F.log_softmax(dist.logits, dim=-1)
                        # select action with current policy

                        # squeeze/unsqueeze is needed because of the attention mechanism
                        # sum is because the attention outputs as many values as hidden dimension
                        qf1_next_tgt = qf1_target(next_obs_vec, rules_emb_next)
                        qf2_next_tgt = qf2_target(next_obs_vec, rules_emb_next)
                        qf1_next_tgt = qf1_next_tgt.mean(1)
                        qf2_next_tgt = qf2_next_tgt.mean(1)

                        min_qf_next_target = (
                            next_pr * (torch.min(qf1_next_tgt, qf2_next_tgt))
                            - alpha * next_state_log_pi
                        )
                        min_qf_next_target = min_qf_next_target.sum(dim=1)
                        r = data["rewards"]
                        d = data["dones"]
                        next_q_value = r + (1 - d) * args.gamma * min_qf_next_target

                    # use Q-values only for the taken actions
                    obs_vec = data["obs_vec"]
                    rules_emb = data["rules_emb"]
                    sel_idxs = data["sel_idxs"].unsqueeze(1).long()

                    if obs_vec.dim() == 2:
                        obs_vec = obs_vec.unsqueeze(1)

                    qf1_values = qf1(obs_vec, rules_emb).mean(1)
                    qf2_values = qf2(obs_vec, rules_emb).mean(1)

                    qf1_a_values = qf1_values.gather(1, sel_idxs).squeeze(-1)
                    qf2_a_values = qf2_values.gather(1, sel_idxs).squeeze(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    # ACTOR training
                    # _, log_pi, action_probs = actor.get_action(data.observations)
                    dist = lang_agent.get_policy_from_embeddings(obs_vec, rules_emb)
                    log_probs = F.log_softmax(dist.logits, dim=-1)

                    with torch.no_grad():
                        qf1_values = qf1(obs_vec, rules_emb).mean(1)
                        qf2_values = qf2(obs_vec, rules_emb).mean(1)
                        min_qf_values = torch.min(qf1_values, qf2_values)

                    # no need for reparameterization, the expectation can be calculated for discrete actions
                    actor_loss = (
                        dist.probs * ((alpha * log_probs) - min_qf_values)
                    ).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        # re-use action probabilities for temperature loss
                        alpha_loss = (
                            dist.probs.detach()
                            * (-log_alpha.exp() * (log_probs + target_entropy).detach())
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.log_frequency == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
                #
                # logging.info(f"SPS: {int(global_step / (time.time() - start_time))}")
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

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

        if global_step % args.ckpt_interval == 0:
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
                "buffer": buffer,
            }
            if args.autotune:
                save_state["log_alpha"] = log_alpha
                save_state["a_optimizer_state"] = a_optimizer.state_dict()
            save_checkpoint(save_state, ckpt_path)

    envs.close()
    writer.close()
