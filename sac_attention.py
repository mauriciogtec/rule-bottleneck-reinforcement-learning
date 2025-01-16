# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import logging
from math import ceil
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

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
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    buffer_size: int = 1024
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 16
    """the batch size of sample from the reply memory"""
    learning_starts: int = 64
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: float | int = 1/32
    """the frequency of training updates"""
    target_network_frequency: int = 128
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
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

    # LLM
    num_rules: int = 3
    """The number of rules for rule-based LLM-only agent"""
    llm: ValidModels = "gpt-4o-mini-huit"
    """the language model to use"""
    embedder_lm: str = "togethercomputer/m2-bert-80M-8k-retrieval"
    """the language model to use for embeddings"""
    embed_dim: int = 768
    """the dimension of the embeddings"""
    hidden_dim: int = 16
    """the hidden dimension of the networks"""

    # Torch compile
    compile_torch: bool = False


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


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    state_dim = envs.single_observation_space[0].shape[0]
    rule_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    num_rules = args.num_rules
    num_actions = envs.single_action_space.n

    # actor = Actor(envs).to(device)
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
    qf1_target = deepcopy(qf1)
    qf2_target = deepcopy(qf2)

    logging.info("--- Actor ---")
    torchsummary.summary(actor)

    logging.info("--- Q-function ---")
    torchsummary.summary(qf1)

    # language agent
    actor_critic = nn.Module()
    actor_critic.actor = actor
    actor_critic.critic = qf1

    lang_agent = RulesSelectorActorCritic(
        actor=actor,
        task_text=envs.envs[0].metadata["task_text"],
        action_space_text=envs.envs[0].metadata["action_space_text"],
        num_rules=num_rules,
        llm=chat_model,
        embededder=embed_model,
        max_rule_combinations=1,
    )

    # compile the models with torch > 2
    if args.compile_torch:
        actor = torch.compile(actor)
        qf1 = torch.compile(qf1)
        qf2 = torch.compile(qf2)
        qf1_target = torch.compile(qf1_target)
        qf2_target = torch.compile(qf2_target)

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

    buffer = buffers.SimpleDictReplayBuffer(args.buffer_size, device=device)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    obs_vec, obs_text = obs

    # convert to torch tensor
    obs_vec = torch.FloatTensor(obs_vec).to(device)

    # expand state space with the rules, i.e., internal states
    with torch.inference_mode():
        outputs, messages = lang_agent.parallel_pipeline(
            obs_text, obs_vec, pre_action_only=True
        )
    rules = [x["rules"] for x in outputs]
    rules_emb = [x["rules_emb"] for x in outputs]
    autoreset = np.zeros(args.num_envs, dtype=bool)

    for global_step in range(args.total_timesteps):
        with torch.inference_mode():
            outputs, messages = lang_agent.parallel_pipeline(
                state_text=obs_text,
                state_vector=obs_vec,
                pre_action_outputs=outputs,
                pre_action_messages=messages,
            )
        actions = [x["action"] for x in outputs]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs_vec, next_obs_text = next_obs
        next_obs_vec = torch.FloatTensor(next_obs_vec).to(device)
        terminations = torch.FloatTensor(terminations).to(device)

        # Get the next rules
        with torch.inference_mode():
            outputs, _ = lang_agent.parallel_pipeline(
                next_obs_text, next_obs_vec, pre_action_only=True
            )
        next_rules = [x["rules"] for x in outputs]
        next_rules_emb = [x["rules_emb"] for x in outputs]
        sel_idxs = [x["sel_idx"] for x in outputs]

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
                sample["dones"] = terminations[j]
                buffer.add(sample)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        autoreset = np.logical_or(terminations, truncations)
        obs_vec, obs_text = next_obs_vec, next_obs_text

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                for _ in range(ceil(1 / args.update_frequency)):
                    data = buffer.sample(args.batch_size)
                    next_obs_vec = data["next_obs_vec"]
                    next_obs_text = data["next_obs_text"]
                    rules_emb_next = data["rules_emb_next"]

                    # CRITIC training
                    with torch.no_grad():
                        dist = lang_agent.get_policy_from_embeddings(
                            next_obs_vec, rules_emb_next
                        )
                        next_pr = torch.nn.functional.softmax(dist.logits, dim=-1)
                        entropy = dist.entropy()
                        # select action with current policy

                        # squeeze/unsqueeze is needed because of the attention mechanism
                        # sum is because the attention outputs as many values as hidden dimension
                        qf1_next_tgt = qf1_target(next_obs_vec.unsqueeze(1), rules_emb_next)
                        qf2_next_tgt = qf2_target(next_obs_vec.unsqueeze(1), rules_emb_next)
                        qf1_next_tgt = qf1_next_tgt.squeeze(1)
                        qf2_next_tgt = qf2_next_tgt.squeeze(1)

                        min_qf_next_target = next_pr * torch.min(qf1_next_tgt, qf2_next_tgt)
                        min_qf_next_target = min_qf_next_target.sum(dim=1) - alpha * entropy
                        r = data["rewards"]
                        d = data["dones"]
                        next_q_value = r + (1 - d) * args.gamma * (min_qf_next_target)

                    # use Q-values only for the taken actions
                    obs_vec = data["obs_vec"]
                    rules_emb = data["rules_emb"]
                    sel_idxs = data["sel_idxs"].unsqueeze(1).long()

                    qf1_values = qf1(obs_vec.unsqueeze(1), rules_emb).squeeze(1)
                    qf2_values = qf2(obs_vec.unsqueeze(1), rules_emb).squeeze(1)

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
                    dist = lang_agent.get_policy_from_embeddings(
                        data["obs_vec"], data["rules_emb"]
                    )
                    log_probs = torch.nn.functional.log_softmax(dist.logits, dim=-1)
                    sel_probs = dist.probs

                    with torch.inference_mode():
                        qf1_values = qf1(obs_vec.unsqueeze(1), rules_emb).squeeze(1)
                        qf2_values = qf2(obs_vec.unsqueeze(1), rules_emb).squeeze(1)
                        min_qf_values = torch.min(qf1_values, qf2_values)

                    # no need for reparameterization, the expectation can be calculated for discrete actions
                    actor_loss = (sel_probs * ((alpha * log_probs) - min_qf_values)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        # re-use action probabilities for temperature loss
                        alpha_loss = (
                            sel_probs.detach()
                            * (-log_alpha.exp() * (log_probs + target_entropy).detach())
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

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

            if global_step % args.log_frequency == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losseps/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                logging.info(f"SPS: {int(global_step / (time.time() - start_time))}")
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

    envs.close()
    writer.close()
