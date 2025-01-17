import logging
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import agents
from agents import ValidAgents
import envs as E  # registers the gym environments during import
from llm_apis import get_llm_api, ValidModels

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import jsonlines


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rulebots"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    log_examples_interval: int = 10
    """the interval to log examples"""
    wandb_save_code: bool = True
    """if toggled, the code will be saved to wandb"""

    # Environment
    env_id: str = "Uganda"
    """the id of the environment"""
    num_envs: int = 4
    """the number of parallel game environments"""
    agent: ValidAgents = "llm_rules_agent"
    """the agent to use"""
    parallel_pipeline: bool = True
    """if toggled, the pipeline will be parallelized"""
    llm: ValidModels = "gpt-4o-mini-huit"
    """the language model to use"""

    # eval
    num_steps: int = 16
    """the number of steps to run in each eval environment per policy rollout"""
    num_episodes: int = 5
    """the number of eval iterations"""

    # Algorithm
    num_rules: int = 3
    """The number of rules for rule-based LLM-only agent"""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args: Args):
    def make_env(env_id, eval=False):
        def thunk():
            if eval:
                env = gym.make(env_id, max_episode_steps=None, T=args.num_steps)
            else:
                env = gym.make(env_id)

            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk

    eval_env_funs = [make_env(args.env_id, eval=True) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(eval_env_funs)
    chat_model = get_llm_api(args.llm)

    set_seed(args.seed)

    t = int(time())
    run_name = f"eval_llm_{args.env_id}__{args.agent}__{args.llm}__{args.exp_name}__{args.seed}__{t}"
    params = vars(args)

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

    if args.agent == "base_agent":
        lang_agent = agents.BaseAgent(
            task_text=envs.metadata["task_text"],
            action_space_text=envs.metadata["action_space_text"],
            llm=chat_model,
        )
    elif args.agent == "llm_rules_agent":
        lang_agent = agents.LLMRulesAgent(
            task_text=envs.metadata["task_text"],
            action_space_text=envs.metadata["action_space_text"],
            num_rules=args.num_rules,
            llm=chat_model,
        )
    elif args.agent == "no_thoughts_agent":
        lang_agent = agents.NoThoughtsAgent(
            task_text=envs.metadata["task_text"],
            action_space_text=envs.metadata["action_space_text"],
            llm=chat_model,
        )
    else:
        raise ValueError(f"Unknown baseline: {args.agent}")

    text_logs_path = f"text_logs/{run_name}.jsonl"
    json_logger_mode = "w"
    os.makedirs(os.path.dirname(text_logs_path), exist_ok=True)
    jsonl_logger = jsonlines.open(text_logs_path, mode=json_logger_mode)

    global_step = 0

    obs, infos = envs.reset()
    _, next_state_text = obs

    _ep_buffer = defaultdict(lambda: [[] for _ in range(args.num_envs)])

    all_mean_rewards = []

    total_episodes = 0
    step = 0
    while total_episodes < args.num_episodes:
        global_step += args.num_envs

        # Get the action and value in parallel
        if args.parallel_pipeline:
            outputs, messages = lang_agent.parallel_pipeline(next_state_text)
        else:
            outputs, messages = lang_agent(next_state_text)

        # Step the environment
        env_actions = [x["action"] for x in outputs]
        obs, env_rewards, terminations, truncations, infos = envs.step(env_actions)
        _, next_state_text = obs

        if args.agent == "llm_rules_agent":
            sel_reward_scores = [x["sel_reward_scores"] for x in outputs]
            sel_rewards = [x["sel_reward"] for x in outputs]

        # accumulate and log the rewards
        for j in range(args.num_envs):
            done_now = terminations[j] or truncations[j]
            if not done_now:
                _ep_buffer["env_rewards"][j].append(env_rewards[j].item())
                if args.agent == "llm_rules_agent":
                    _ep_buffer["sel_rewards_scores"][j].append(sel_reward_scores[j])
                    _ep_buffer["sel_rewards_total"][j].append(sel_rewards[j])
                    _ep_buffer["total_rewards"][j].append(env_rewards[j].item())
            else:
                mean_reward = np.mean(_ep_buffer["env_rewards"][j])
                all_mean_rewards.append(mean_reward)
                # log the rewards
                writer.add_scalar(
                    f"charts/episodic_env_rewards",
                    mean_reward,
                    global_step,
                )
                _ep_buffer["env_rewards"][j].clear()

                if args.agent == "llm_rules_agent":
                    m = np.mean(_ep_buffer["sel_rewards_scores"][j], axis=0)
                    for i, x in enumerate(m):
                        writer.add_scalar(
                            f"charts/sel_reward_scores/q{i}", x, global_step
                        )
                    m = np.mean(_ep_buffer["sel_rewards_total"][j])
                    writer.add_scalar("charts/episodic_sel_rewards", m, global_step)
                    writer.add_scalar(
                        "charts/episodic_total_rewards",
                        np.sum(_ep_buffer["total_rewards"][j]),
                        global_step,
                    )
                    _ep_buffer["sel_rewards_scores"][j].clear()
                    _ep_buffer["sel_rewards_total"][j].clear()
                    _ep_buffer["total_rewards"][j].clear()

                total_episodes += 1

        if "episode" in infos:
            for i in range(args.num_envs):
                if infos["_episode"][i]:
                    r, l = infos["episode"]["r"][i], infos["episode"]["l"][i]
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)

                    logging.info(f"global_step={global_step}, episodic_return={r:.4f}")

        if step == 0 or step % args.log_examples_interval == 0:
            if args.agent == "llm_rules_agent":
                rules_str = "\n".join(outputs[0]["rules"])
                rules_scores = [
                    f"{k}: {v}" for k, v in outputs[0]["sel_reward_scores_raw"].items()
                ]
                rules_scores_str = "\n".join(rules_scores)
                example = (
                    f"{outputs[0]['initial_prompt']}\n"
                    f"### Thoughts\n {outputs[0]['thoughts']}\n"
                    f"### Rules\n {rules_str}\n"
                    f"### Selected Rules Explainability\n{rules_scores_str}\n"
                    f"### Environment Action\[0]n {env_actions}\n"
                    f"### Explanation\n {outputs[0]['explanation']}\n"
                    f"### Explanation rules only\n {outputs[0]['explanation_rule_only']}"
                )
            elif args.agent == "no_thoughts_agent":
                example = (
                    f"{outputs[0]['initial_prompt']}\n"
                    f"### Environment Action\[0]n {env_actions}\n"
                    f"### Explanation\n {outputs[0]['explanation']}"
                )
            elif args.agent == "base_agent":
                example = (
                    f"{outputs[0]['initial_prompt']}\n"
                    f"### Thoughts\n {outputs[0]['thoughts']}\n"
                    f"### Environment Action\[0]n {env_actions}\n"
                    f"### Explanation\n {outputs[0]['explanation']}"
                    f"### Explanation no thoughts\n {outputs[0]['explanation_no_thoughts']}"
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

        if step % 64 == 0:
            logging.info(
                f"global_step={global_step}, total_episodes={total_episodes}/{args.num_episodes}"
            )
        step += 1

    # Log summary statistics
    mean_reward = np.mean(all_mean_rewards)
    std_reward = np.std(all_mean_rewards)
    writer.add_scalar("mean_reward", mean_reward, 0)
    writer.add_scalar("std_reward", std_reward, 0)

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
