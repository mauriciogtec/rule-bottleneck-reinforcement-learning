# this file is an adaption of the clean rl's PPO implementation for rule-based agents
# rule-based language agents use different internal actions than the environment actions
# in addition, one must keep track of the rules selected by the agent.

from dataclasses import dataclass
import itertools
import logging
import shutil
import os
import random
import tyro
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torchsummary
from langchain_together import TogetherEmbeddings
from torch.nested import nested_tensor
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
from torch import tensor, FloatTensor

from llm_apis import get_llm_api, ValidLLMs
import envs as E  # registers the gym environments during import
from agents import RulesSelectorActorCritic
from layers import CrossAttentionNetwork, pool_attention_network

# configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


import jsonlines


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, checkpoint_path):
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
        return None


def collate_samples(x: List[List]):
    """Takes as input a list of lists where the first index is the environment index
    and the second one is the time step index"""
    # first concatenate the lists using itertools
    x = list(itertools.chain(*x))

    # now check if all the first shapes are the same
    same_shape = all(x[0].shape == y.shape for y in x)
    if same_shape:
        return torch.stack(x)
    else:
        return nested_tensor(x)


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
    log_examples_interval: int = 20
    """the interval to log examples"""
    save_interval: int = 1
    """the interval to save the model"""
    resume: bool = False
    """if toggled, the model will be resumed from the last checkpoint"""
    max_rule_combinations: int = 1
    """the maximum number of rule combinations to use"""
    exp_name: str = ""
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    ckpt_interval: int = 1
    """the saving interval of the model"""
    overwrite_ckpt: bool = False
    """if toggled and resuming is on, it will start fresh in resume mode, otherwise ignored"""

    # Logging and tracking arguments
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rulebots"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_save_code: bool = True
    """if toggled, the code will be saved to wandb"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    log_examples_interval: int = 20
    """the interval to log examples"""
    save_interval: int = 1
    """the interval to save the model"""

    # PPO specific arguments
    env_id: str = "Uganda"
    """the id of the environment"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 32
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    dropout: float = 0.0
    """the dropout rate"""

    num_eval_steps: int = 64
    """the number of steps to run in each eval environment per policy rollout"""
    eval_interval: int = 4
    """the evaluation interval"""
    eval_deterministic: bool = True
    """if toggled, the evaluation will be deterministic"""
    rolling_returns_window: int = 64
    """the rolling rewards window"""

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
    parallel_pipeline: bool = True
    """if toggled, the pipeline will be parallelized"""


def main(args: Args):
    run_id = f"ppo_attention_{args.env_id}__{args.exp_name}__{args.llm}__{args.seed}"
    run_name = run_id if args.resume else f"{run_id}__{int(time.time())}"

    ckpt_path = f"checkpoints/{run_name}.state"
    text_logs_path = f"text_logs/{run_name}.jsonl"
    json_logger_mode = "w" if not args.resume else "a"
    os.makedirs(os.path.dirname(text_logs_path), exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    jsonl_logger = jsonlines.open(text_logs_path, mode=json_logger_mode)

    if args.overwrite_ckpt:
        # delete checkpoint path
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        # delete tensorboard logs
        dirname = f"runs/{run_name}"
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

    # train_env_funs = [
    #     make_env(args.env_id, args.seed + i) for i in range(args.num_envs)
    # ]x
    # eval_env_funs = [
    #     make_env(args.env_id, 1000 * args.seed + i, eval=True)
    #     for i in range(args.num_envs)
    # ]

    wrappers = [gym.wrappers.RecordEpisodeStatistics]
    envs = gym.make_vec(args.env_id, args.num_envs, wrappers=wrappers)
    eval_envs = gym.make_vec(args.env_id, args.num_envs, wrappers=wrappers)

    chat_model = get_llm_api(args.llm)
    embed_model = TogetherEmbeddings(model=args.embedder_lm)

    state_dim = envs.single_observation_space[0].shape[-1]
    rule_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    num_rules = args.num_rules

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

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

    actor = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
    )
    critic = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
    )
    model = nn.ModuleDict({"actor": actor, "critic": critic}).to(dev)

    example_rules = envs.envs[0].metadata["example_rules"]
    example_rules = "".join(f"- {x}\n" for x in example_rules)
    lang_agent = RulesSelectorActorCritic(
        actor=actor,
        task_text=envs.envs[0].metadata["task_text"],
        action_space_text=envs.envs[0].metadata["action_space_text"],
        num_rules=num_rules,
        llm=chat_model,
        embededder=embed_model,
        max_rule_combinations=1,
        example_rules=example_rules,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    logging.info("=== Actor ===")
    torchsummary.summary(actor)

    logging.info("=== Critic ===")
    torchsummary.summary(critic)

    global_step = 0
    start_time = time.time()
    best_total_reward = -float("inf")
    best_model = None

    checkpoint = load_checkpoint(ckpt_path, dev)
    if args.resume and checkpoint:
        logging.info(
            f"Resuming training from checkpoint at step {checkpoint['global_step']}."
        )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        global_step = checkpoint["global_step"]
        start_time = time.time() - checkpoint["elapsed_time"]
        best_total_reward = checkpoint["best_total_reward"]
        best_model = checkpoint["best_model"]
        logging.info(f"Resumed training from checkpoint at step {global_step}.")

    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_iterations = args.total_timesteps // batch_size

    obs, infos = envs.reset()
    state_vector, state_text = obs
    state_vector = tensor(state_vector, dtype=torch.float32).to(dev)
    dones = [False for _ in range(args.num_envs)]

    iter_start = (global_step // args.total_timesteps) + 1

    # keep logging buffers for the rewards
    autoreset = np.zeros(args.num_envs, dtype=bool)
    _ep_buffer = defaultdict(lambda: [[] for _ in range(args.num_envs)])
    _rolling_returns = deque(maxlen=args.rolling_returns_window)

    for iter in range(iter_start, num_iterations + 1):
        # start on policy buffer
        b_state_vec = [[] for _ in range(args.num_envs)]
        b_done = [[] for _ in range(args.num_envs)]
        b_rules_emb = [[] for _ in range(args.num_envs)]
        b_sel_idx = [[] for _ in range(args.num_envs)]
        b_value = [[] for _ in range(args.num_envs)]
        b_sel_logprob = [[] for _ in range(args.num_envs)]
        b_reward = [[] for _ in range(args.num_envs)]

        if args.anneal_lr:
            frac = 1.0 - (iter - 1.0) / num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in tqdm(
            range(args.num_steps), desc=f"Iter: {iter},  Gathering trajectories"
        ):
            global_step += 1  # args.num_envs # changed to 1 to align with sac

            # Get the action and value in parallel
            if args.parallel_pipeline:
                outputs, messages = lang_agent.parallel_pipeline(
                    state_text,
                    state_vector,
                )
            else:
                outputs, messages = lang_agent(state_text, state_vector)

            # Add states and dones
            actions = [x["action"] for x in outputs]
            entropies = [x["entropy"] for x in outputs]
            sel_logprobs = [x["sel_logprob"] for x in outputs]

            # Step the environment
            next_obs, env_rewards, dones, trunc, infos = envs.step(actions)
            next_state_vector, next_state_text = next_obs
            next_state_vector = tensor(next_state_vector, dtype=torch.float32)
            next_state_vector = next_state_vector.to(dev)

            sel_rewards = [x["sel_reward"] for x in outputs]
            sel_reward_scores = [x["sel_reward_scores"] for x in outputs]
            # sel_reward_scores_raw = [x["sel_reward_scores_raw"] for x in outputs]

            # total rewards
            rewards = env_rewards + sel_rewards
            _rolling_returns.extend(list(rewards))

            # compute the values
            # mean pool over rules for state-value
            rules_emb = [FloatTensor(x["rules_emb"]) for x in outputs]
            rules_emb = nested_tensor(rules_emb).to(dev)
            values = critic(rules_emb, state_vector)
            values = pool_attention_network(values)

            # add the transition to the buffer
            for j in range(args.num_envs):
                if not autoreset[j]:
                    b_state_vec[j].append(state_vector[j])
                    b_rules_emb[j].append(outputs[j]["rules_emb"])
                    b_done[j].append(tensor(dones[j], dtype=torch.float32).to(dev))
                    b_sel_idx[j].append(outputs[j]["sel_idx"])
                    b_sel_logprob[j].append(outputs[j]["sel_logprob"])
                    b_reward[j].append(tensor(rewards[j], dtype=torch.float32).to(dev))
                    b_value[j].append(tensor(values[j], dtype=torch.float32).to(dev))

            # accumulate and log the rewards
            for j in range(args.num_envs):
                needs_reset = trunc[j] or dones[j]
                if not needs_reset:
                    _ep_buffer["env_rewards"][j].append(env_rewards[j].item())
                    _ep_buffer["sel_rewards_scores"][j].append(sel_reward_scores[j])
                    _ep_buffer["sel_rewards_total"][j].append(sel_rewards[j].item())
                    _ep_buffer["total_rewards"][j].append(rewards[j].item())
                    _ep_buffer["entropy"][j].append(entropies[j].item())
                    _ep_buffer["sel_probs"][j].append(sel_logprobs[j].exp().item())
                else:
                    # log the rewards
                    writer.add_scalar(
                        f"charts/episodic_env_rewards",
                        np.mean(_ep_buffer["env_rewards"][j]).item(),
                        global_step,
                    )
                    m = np.mean(_ep_buffer["sel_rewards_scores"][j], axis=0)
                    for i, x in enumerate(m):
                        writer.add_scalar(
                            f"charts/sel_reward_scores/q{i}", x.item(), global_step
                        )
                    m = np.mean(_ep_buffer["sel_rewards_total"][j])
                    writer.add_scalar(
                        "charts/episodic_sel_rewards", m.item(), global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_entropy",
                        np.mean(_ep_buffer["entropy"][j]).item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_sel_probs",
                        np.mean(_ep_buffer["sel_probs"][j]).item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_total_rewards",
                        np.mean(_ep_buffer["total_rewards"][j]).item(),
                        global_step,
                    )

                    # flush
                    _ep_buffer["env_rewards"][j].clear()
                    _ep_buffer["sel_rewards_scores"][j].clear()
                    _ep_buffer["sel_rewards_total"][j].clear()
                    _ep_buffer["entropy"][j].clear()
                    _ep_buffer["sel_probs"][j].clear()
                    _ep_buffer["total_rewards"][j].clear()

            if "episode" in infos:
                for i in range(args.num_envs):
                    if infos["_episode"][i]:
                        r, l = infos["episode"]["r"][i], infos["episode"]["l"][i]
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)

                        logging.info(
                            f"global_step={global_step}, episodic_return={r:.4f}"
                        )

            if step == 0 or step % args.log_examples_interval == 0:
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

            # advance
            autoreset = np.logical_or(autoreset, dones)
            obs = next_obs
            state_vector = next_state_vector
            state_text = next_state_text

            # save best model
            total_reward = np.mean(_rolling_returns)
            if (
                best_total_reward < total_reward
                and len(_rolling_returns) == args.rolling_returns_window
            ):
                best_total_reward = total_reward
                best_model = model.state_dict()

        if iter % args.save_interval == 0 or iter == num_iterations:
            ckpt = {
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "global_step": global_step,
                "elapsed_time": time.time() - start_time,
                "best_total_reward": best_total_reward,
                "best_model": best_model,
            }
            torch.save(ckpt, ckpt_path)

        # need to obtain the new rules
        if args.parallel_pipeline:
            outputs, messages = lang_agent.parallel_pipeline(
                next_state_text, next_state_vector, pre_action_only=True
            )
        else:
            outputs, messages = lang_agent(
                next_state_text, next_state_vector, pre_action_only=True
            )

        next_rules_emb = [FloatTensor(x["rules_emb"]) for x in outputs]
        next_rules_emb = nested_tensor(next_rules_emb).to(dev)
        dones = FloatTensor(dones).to(dev)

        # bootstrap value if not done
        with torch.no_grad():
            next_values = critic(next_rules_emb, next_state_vector)
            next_values = pool_attention_network(next_values)

            # compute the advantages
            # need to do per environment to deal with potentially different lengths
            # as well as nested tensors for the rules
            # (and maybe future support for different state space shapes)
            b_advantage = []
            b_return = []

            for j in range(args.num_envs):
                # it is not equal to num steps due to the autoreset skip
                T = len(b_reward[j])
                b_advantage.append(torch.zeros(T, device=dev))
                b_return.append(torch.zeros(T, device=dev))

                lastgaelam = 0
                for t in reversed(range(T)):
                    if t == T - 1:
                        nextnonterminal = 1.0 - dones[j]
                        nextvalues = next_values[j]
                    else:
                        nextnonterminal = 1.0 - b_done[j][t + 1]
                        nextvalues = b_value[j][t + 1]
                    delta = (
                        b_reward[j][t]
                        + args.gamma * nextvalues * nextnonterminal
                        - b_value[j][t]
                    )
                    b_advantage[j][t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                    b_return[j][t] = b_advantage[j][t] + b_value[j][t]

        # flatten the batch
        b_state_vec = collate_samples(b_state_vec)
        b_sel_logprob = collate_samples(b_sel_logprob)
        b_sel_idx = collate_samples(b_sel_idx)
        b_advantage = collate_samples(b_advantage)
        b_return = collate_samples(b_return)
        b_value = collate_samples(b_value)
        b_rules_emb = collate_samples(b_rules_emb)

        # Optimizing the policy and value network
        N = b_state_vec.shape[0]
        b_inds = torch.arange(N, dtype=torch.long).to(dev)
        clipfracs = []
        for epoch in tqdm(range(args.update_epochs), desc=f"Iter: {iter},  Optimizing"):
            b_inds = torch.randperm(N).to(dev)
            for start in range(0, N, minibatch_size):
                end = min(start + minibatch_size, N)
                mb_inds = b_inds[start:end]

                # get the new policy
                mb_rules_emb = nested_tensor([b_rules_emb[ix] for ix in mb_inds]).to(
                    dev
                )
                dist = lang_agent.get_policy_from_embeddings(
                    b_state_vec[mb_inds],
                    mb_rules_emb,
                )
                new_sel_logprob = dist.log_prob(b_sel_idx[mb_inds])
                new_entropy = dist.entropy().mean()

                # get the new value
                new_value = critic(mb_rules_emb, b_state_vec[mb_inds])
                new_value = pool_attention_network(new_value)

                logratio = new_sel_logprob - b_sel_logprob[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantage[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = (new_value - b_return[mb_inds]) ** 2
                    v_clipped = b_value[mb_inds] + torch.clamp(
                        new_value - b_value[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_return[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_value - b_return[mb_inds]) ** 2).mean()

                entropy_loss = new_entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_value.cpu().numpy(), b_return.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # # Run eval episodes
        # if iter % args.eval_interval == 0 or iter == num_iterations:
        #     eval_rewards = []
        #     sel_rewards = []
        #     obs, _ = eval_envs.reset(seed=args.seed)
        #     state_vector, state_text = obs
        #     state_vector = tensor(state_vector, dtype=torch.float32).to(device)
        #     for _ in tqdm(
        #         range(args.num_eval_steps), desc=f"Iter: {iter},  Evaluating"
        #     ):
        #         with torch.no_grad():
        #             if args.parallel_pipeline:
        #                 actions, outputs, _ = parallel_pipeline(
        #                     lang_agent, state_text, state_vector, post_action=True
        #                 )
        #             else:
        #                 actions, outputs, _ = lang_agent(
        #                     state_text, state_vector, post_action=True
        #                 )
        #         obs, reward, _, _, infos = eval_envs.step(actions)
        #         state_vector, state_text = obs
        #         state_vector = FloatTensor(state_vector).to(device)
        #         sel_rewards.append(FloatTensor(outputs["sel_reward"]).to(device))
        #         eval_rewards.append(FloatTensor(reward).to(device))
        #     eval_rewards = torch.stack(eval_rewards).mean().item()
        #     sel_rewards = torch.stack(sel_rewards).mean().item()
        #     total_rewards = eval_rewards + sel_rewards
        #     writer.add_scalar("charts/eval_episodic_return", eval_rewards, global_step)
        #     writer.add_scalar("charts/eval_sel_rule_return", sel_rewards, global_step)
        #     writer.add_scalar("charts/eval_total_return", total_rewards, global_step)

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.parse(Args)
    main(args)
