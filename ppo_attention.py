# this file is an adaption of the clean rl's PPO implementation for rule-based agents
# rule-based language agents use different internal actions than the environment actions
# in addition, one must keep track of the rules selected by the agent.

import itertools
import logging
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.optim as optim
import torchsummary
from langchain_together import ChatTogether, TogetherEmbeddings
from omegaconf import DictConfig, OmegaConf
from torch.nested import nested_tensor, to_padded_tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from apis import HUITMistralModel
import envs as E  # registers the gym environments during import
from agents import RulesSelectorActorCritic
from layers import AttentionActorCritic

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import jsonlines

ValidModels = Literal[
    "google/gemma-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
]

ModelDict = {
    "google/gemma-2b-it": ChatTogether,
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": ChatTogether,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ChatTogether,
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ChatTogether,
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ChatTogether,
    "meta.llama3-1-8b-instruct-v1:0": HUITMistralModel,
    "meta.llama3-1-70b-instruct-v1:0": HUITMistralModel,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def pad_rules(rules_emb: list[list[torch.Tensor]]):
    num_steps = len(rules_emb)
    num_envs = len(rules_emb[0])

    x = list(itertools.chain(*rules_emb))
    embs = nested_tensor(x)
    padding_mask = nested_tensor([torch.ones(len(y)) for y in x])

    embs = to_padded_tensor(embs, -20.0)
    padding_mask = to_padded_tensor(padding_mask, 0.0)

    embs = embs.view(num_steps, num_envs, -1, embs.shape[-1])
    padding_mask = padding_mask.view(num_steps, num_envs, -1)

    return embs, padding_mask


def save_checkpoint(state, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
        return None


def parallel_pipeline(lang_agent, state_text, state_vector, post_action=True):
    # Get the action and value in parallel
    def call_pipeline(i):
        with torch.no_grad():
            return lang_agent(
                state_text=state_text[i],
                state_vector=state_vector[i],
                post_action=post_action,
            )

    num_envs = len(state_text)
    num_procs = min(os.cpu_count() - 1, num_envs)
    with ThreadPoolExecutor(max_workers=num_procs) as executor:
        results = list(executor.map(call_pipeline, range(num_envs)))
        env_actions, outputs, messages = zip(*results)
        env_actions = list(env_actions)
        outputs = {key: [output[key] for output in outputs] for key in outputs[0]}
        messages = list(messages)

    return env_actions, outputs, messages


@hydra.main(config_path="conf", config_name="ppo_attention", version_base=None)
def main(args: DictConfig):
    def make_env(env_id, eval=False):
        def symlog(x: float) -> float:
            import math

            return math.copysign(math.log1p(abs(x)), x)

        def thunk():
            if eval:
                env = gym.make(env_id, max_episode_steps=None, T=args.num_eval_steps)
            else:
                env = gym.make(env_id)

            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.TransformReward(env, symlog)
            return env

        return thunk

    train_env_funs = [make_env(args.env_id) for i in range(args.num_envs)]
    eval_env_funs = [make_env(args.env_id, eval=True) for i in range(args.num_envs)]
    envs = (
        gym.vector.AsyncVectorEnv(train_env_funs, shared_memory=False)
        if args.parallel_envs
        else gym.vector.SyncVectorEnv(train_env_funs)
    )
    eval_envs = (
        gym.vector.AsyncVectorEnv(eval_env_funs, shared_memory=False)
        if args.parallel_envs
        else gym.vector.SyncVectorEnv(eval_env_funs)
    )

    chat_model = ModelDict[args.chat_lm](model=args.chat_lm)
    embed_model = TogetherEmbeddings(model=args.embedder_lm)

    state_dim = envs.single_observation_space[0].shape[0]
    rule_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    num_rules = args.num_rules

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_id = f"ppo_attention_{args.env_id}__{args.exp_name}__{args.seed}"

    run_name = run_id if args.resume else f"{run_id}_{int(time.time())}"
    params = OmegaConf.to_container(args, resolve=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=params,
            name=run_name,
            monitor_gym=True,
            save_code=args.wandb_save_code,
        )
    writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in params.items()])),
    )

    actor_critic = AttentionActorCritic(state_dim, rule_dim, hidden_dim).to(device)
    lang_agent = RulesSelectorActorCritic(
        actor_critic=actor_critic,
        task_text=envs.metadata["task_text"],
        action_space_text=envs.metadata["action_space_text"],
        llm=chat_model,
        embededder=embed_model,
        num_rules=num_rules,
        max_rule_combinations=args.max_rule_combinations,
    )
    optimizer = optim.Adam(actor_critic.parameters(), lr=args.learning_rate, eps=1e-5)
    torchsummary.summary(actor_critic)

    ckpt_path = f"checkpoints/ppo_attention/best_{run_id}.state"
    text_logs_path = f"text_logs/ppo_attention/{run_id}.jsonl"
    json_logger_mode = "w" if not args.resume else "a"
    jsonl_logger = jsonlines.open(text_logs_path, mode=json_logger_mode)

    global_step = 0
    start_time = time.time()
    best_total_reward = -float("inf")
    best_model = None

    checkpoint = load_checkpoint(ckpt_path, device)
    if args.resume and checkpoint:
        logger.info(
            f"Resuming training from checkpoint at step {checkpoint['global_step']}."
        )
        actor_critic.load_state_dict(checkpoint["model_state"])
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
    next_state_vector, next_state_text = obs
    next_state_vector = torch.tensor(next_state_vector, dtype=torch.float32).to(device)
    next_dones = torch.zeros(args.num_envs, dtype=torch.bool).to(device)

    # TODO: include iteration in the checkpoint
    # deduce first iteration from global step
    iter_start = (global_step // args.total_timesteps) + 1
    for iter in range(iter_start, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iter - 1.0) / num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        transitions = defaultdict(list)
        for step in tqdm(
            range(args.num_steps), desc=f"Iter: {iter},  Gathering trajectories"
        ):
            global_step += args.num_envs

            # Add states and dones
            transitions["state_vector"].append(next_state_vector)
            transitions["state_text"].append(next_state_text)
            transitions["dones"].append(next_dones)

            # Get the action and value in parallel
            if args.parallel_pipeline:
                env_actions, outputs, messages = parallel_pipeline(
                    lang_agent, next_state_text, next_state_vector, post_action=True
                )
            else:
                env_actions, outputs, messages = lang_agent(
                    next_state_text, next_state_vector, post_action=True
                )

            # Append the rules
            transitions["rules"].append(outputs["rules"])
            transitions["rules_emb"].append(outputs["rules_emb"])

            # Append the scalar quantities
            for key in ["sel_idx", "sel_logprob", "value", "entropy", "sel_reward"]:
                transitions[key].append(torch.stack(outputs[key]))

            # Step the environment
            obs, rewards, terminated, truncated, infos = envs.step(env_actions)
            next_state_vector, next_state_text = obs
            next_state_vector = torch.tensor(next_state_vector, dtype=torch.float32)
            next_state_vector = next_state_vector.to(device)
            next_dones = torch.FloatTensor(terminated | truncated).to(device)

            # Store the reward and explainability scores
            transitions["rewards"].append(list(rewards))
            transitions["sel_reward_scores_raw"].append(
                outputs["sel_reward_scores_raw"]
            )
            transitions["sel_reward_scores"].append(outputs["sel_reward_scores"])

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
                rules_str = "\n".join(outputs["rules"][0])
                rules_scores = [
                    f"{k}: {v}" for k, v in outputs["sel_reward_scores_raw"][0].items()
                ]
                rules_scores_str = "\n".join(rules_scores)
                example = (
                    f"{outputs['initial_prompt'][0]}\n"
                    f"### Thoughts\n{outputs['thoughts'][0]}\n"
                    f"### Rules\n{rules_str}\n"
                    f"### Selected Rule\n{outputs['sel_rule'][0]}\n"
                    f"### Selected Rule Probability\n{outputs['sel_logprob'][0].exp():.2f}\n"
                    f"### Selected Rule Reward\n {outputs['sel_reward'][0]}\n"
                    f"### Selected Rule Explainability\n{rules_scores_str}\n"
                    f"### Environment Action\n{env_actions[0]}\n"
                    f"### Explanation with thoughts and all rules\n{outputs['explanation'][0]}\n"
                    f"### Explanation with only selected rule\n{outputs['explanation_rule_only'][0]}"
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
                        "conversation": conversation,
                    }
                )

        # convert the transitions to tensors
        state_vector = torch.stack(transitions["state_vector"])
        rules_emb, rules_padding_mask = pad_rules(transitions["rules_emb"])
        dones = torch.stack(transitions["dones"])
        values = torch.stack(transitions["value"])
        logprobs = torch.stack(transitions["sel_logprob"])
        sel_idxs = torch.stack(transitions["sel_idx"])
        env_rewards = torch.FloatTensor(transitions["rewards"]).to(device)
        sel_rule_rewards = torch.stack(transitions["sel_reward"])
        rewards = sel_rule_rewards + env_rewards

        sel_reward_scores = np.array(transitions["sel_reward_scores"])

        # save best model
        total_reward = rewards.mean().item()
        if best_total_reward < total_reward:
            best_total_reward = total_reward
            best_model = actor_critic.state_dict()

        if iter % args.save_interval == 0 or iter == num_iterations:
            save_checkpoint(
                {
                    "model_state": actor_critic.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "global_step": global_step,
                    "elapsed_time": time.time() - start_time,
                    "best_total_reward": best_total_reward,
                    "best_model": best_model,
                },
                ckpt_path,
            )

        # bootstrap value if not done
        with torch.no_grad():
            if args.parallel_pipeline:
                env_actions, outputs, messages = parallel_pipeline(
                    lang_agent, next_state_text, next_state_vector, post_action=True
                )
            else:
                env_actions, outputs, messages = lang_agent(
                    next_state_text, next_state_vector, post_action=True
                )

            next_value = torch.stack(outputs["value"])

            # compute the advantages
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_dones.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_states_vec = state_vector.reshape(
            (-1,) + envs.single_observation_space[0].shape
        )
        b_logprobs = logprobs.reshape(-1)
        b_sel_idxs = sel_idxs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rules_emb = rules_emb.reshape((-1,) + rules_emb.shape[2:])
        b_padding_mask = rules_padding_mask.reshape(
            (-1,) + rules_padding_mask.shape[2:]
        )

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in tqdm(range(args.update_epochs), desc=f"Iter: {iter},  Optimizing"):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = (
                    lang_agent.get_action_and_value_from_embeddings(
                        b_states_vec[mb_inds],
                        b_rules_emb[mb_inds],
                        rules_padding_mask=b_padding_mask[mb_inds],
                        sel_idxs=b_sel_idxs[mb_inds],
                    )
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
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
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor_critic.parameters(), args.max_grad_norm
                )
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
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

        total_reward = rewards.mean().item()
        writer.add_scalar("charts/total_reward", total_reward, global_step)

        env_rewards = env_rewards.mean().item()
        sel_rule_rewards = sel_rule_rewards.mean().item()
        writer.add_scalar("charts/env_return", env_rewards, global_step)
        writer.add_scalar("charts/sel_rule_return", sel_rule_rewards, global_step)

        # log sel_reward_scores
        # get means for but the last dimension
        sel_reward_scores = np.mean(sel_reward_scores, axis=(0, 1))
        for i, x in enumerate(sel_reward_scores):
            writer.add_scalar(f"charts/sel_reward_scores/q{i}", x, global_step)

        # Run eval episodes
        if iter % args.eval_interval == 0 or iter == num_iterations:
            eval_rewards = []
            sel_rewards = []
            obs, _ = eval_envs.reset(seed=args.seed)
            state_vector, state_text = obs
            state_vector = torch.tensor(state_vector, dtype=torch.float32).to(device)
            for _ in tqdm(
                range(args.num_eval_steps), desc=f"Iter: {iter},  Evaluating"
            ):
                with torch.no_grad():
                    if args.parallel_pipeline:
                        actions, outputs, _ = parallel_pipeline(
                            lang_agent, state_text, state_vector, post_action=True
                        )
                    else:
                        actions, outputs, _ = lang_agent(
                            state_text, state_vector, post_action=True
                        )
                obs, reward, _, _, infos = eval_envs.step(actions)
                state_vector, state_text = obs
                state_vector = torch.FloatTensor(state_vector).to(device)
                sel_rewards.append(torch.FloatTensor(outputs["sel_reward"]).to(device))
                eval_rewards.append(torch.FloatTensor(reward).to(device))
            eval_rewards = torch.stack(eval_rewards).mean().item()
            sel_rewards = torch.stack(sel_rewards).mean().item()
            total_rewards = eval_rewards + sel_rewards
            writer.add_scalar("charts/eval_episodic_return", eval_rewards, global_step)
            writer.add_scalar("charts/eval_sel_rule_return", sel_rewards, global_step)
            writer.add_scalar("charts/eval_total_return", total_rewards, global_step)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
