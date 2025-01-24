import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from time import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import agents
from agents import PureLanguageAgents
import envs as E  # registers the gym environments during import
from llm_apis import get_llm_api, ValidLLMs

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
    agent: PureLanguageAgents = "llm_rules_no_thoughts"
    """the agent to use"""
    parallel_pipeline: bool = True
    """if toggled, the pipeline will be parallelized"""
    llm: ValidLLMs = "gpt-4o-mini-huit"
    """the language model to use"""
    use_thoughts_with_rules: bool = False
    """if toggled, the thoughts will be used with rules"""

    # eval
    num_episodes: int = 16
    """the number of eval iterations"""
    max_episode_steps: int = 64
    """the number of steps per eval iteration"""

    # Algorithm
    num_rules: int = 1
    """The number of rules for rule-based LLM-only agent"""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_env(env_id, seed, max_episode_steps=None):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env

    return thunk


def main(args: Args):
    eval_env_funs = [
        make_env(args.env_id, args.seed + i, max_episode_steps=args.max_episode_steps)
        for i in range(args.num_envs)
    ]
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
        example_rules = envs.envs[0].metadata["example_rules"]
        example_rules = "".join(f"- {x}\n" for x in example_rules)
        lang_agent = agents.LLMRulesAgent(
            task_text=envs.metadata["task_text"],
            action_space_text=envs.metadata["action_space_text"],
            num_rules=args.num_rules,
            llm=chat_model,
            example_rules=example_rules,
        )
    elif args.agent == "llm_rules_no_thoughts":
        example_rules = envs.envs[0].metadata["example_rules"]
        example_rules = "".join(f"- {x}\n" for x in example_rules)
        lang_agent = agents.LLMRulesAgentNoThoughts(
            task_text=envs.metadata["task_text"],
            action_space_text=envs.metadata["action_space_text"],
            num_rules=args.num_rules,
            llm=chat_model,
            example_rules=example_rules,
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
    all_returns = []

    total_episodes = 0
    step = 0
    autoreset = np.zeros(args.num_envs, dtype=bool)

    while total_episodes < args.num_episodes:
        global_step += 1

        # Get the action and value in parallel
        if args.parallel_pipeline:
            outputs, messages = lang_agent.parallel_pipeline(next_state_text)
        else:
            outputs, messages = lang_agent(next_state_text)
        env_actions = [x["action"] for x in outputs]

        # Step the environment
        obs, env_rewards, terminations, truncations, infos = envs.step(env_actions)
        _, next_state_text = obs

        if args.agent == "llm_rules_agent":
            sel_reward_scores = [x["sel_reward_scores"] for x in outputs]
            sel_rewards = [x["sel_reward"] for x in outputs]

        if "episode" in infos:
            for i in range(args.num_envs):
                if infos["_episode"][i]:
                    r, l = infos["episode"]["r"][i], infos["episode"]["l"][i]
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)

                    all_returns.append(r)

                    logging.info(f"global_step={global_step}, episodic_return={r:.4f}")

        # accumulate and log the rewards
        for j in range(args.num_envs):
            done_now = terminations[j] or truncations[j]
            if not done_now and not autoreset[j]:
                _ep_buffer["env_rewards"][j].append(env_rewards[j])
                if args.agent == "llm_rules_agent":
                    _ep_buffer["sel_rewards_scores"][j].append(sel_reward_scores[j])
                    _ep_buffer["sel_rewards_total"][j].append(sel_rewards[j])
                    _ep_buffer["total_rewards"][j].append(
                        env_rewards[j] + sel_rewards[j]
                    )
            elif not autoreset[j]:
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
                        np.mean(_ep_buffer["total_rewards"][j]),
                        global_step,
                    )
                    _ep_buffer["sel_rewards_scores"][j].clear()
                    _ep_buffer["sel_rewards_total"][j].clear()
                    _ep_buffer["total_rewards"][j].clear()

                total_episodes += 1

        if step == 0 or step % args.log_examples_interval == 0:
            if args.agent in ("llm_rules_agent", "llm_rules_no_thoughts"):
                rules_str = "\n".join(outputs[0]["rules"])
                rules_scores = [
                    f"{k}: {v}"
                    for k, v in outputs[0]["sel_reward_scores_raw"].items()
                ]
                rules_scores_str = "\n".join(rules_scores)
                thoughts = outputs[0].get("thoughts", None)
                example = [
                    f"{outputs[0]['initial_prompt']}\n",
                    f"### Thoughts\n {thoughts}\n" if thoughts else "",
                    f"### Rules\n {rules_str}\n",
                    f"### Selected Rules Explainability\n{rules_scores_str}\n",
                    f"### Environment Action {outputs[0]['action']}\n",
                    f"### Explanation\n {outputs[0]['explanation']}\n",
                ]
                example = "".join(example)
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

        if step % 16 == 0:
            logging.info(
                f"global_step={global_step}, total_episodes={total_episodes}/{args.num_episodes}"
            )
        step += 1
        autoreset = np.array(terminations) | np.array(truncations)

    # Log summary statistics
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    writer.add_scalar("mean_return", mean_return, 0)
    writer.add_scalar("std_return", std_return, 0)
    print(f"mean_reward: {mean_return}, std_reward: {std_return}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
