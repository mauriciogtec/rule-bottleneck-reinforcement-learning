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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rulebots-eval-llm"
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
    num_envs: int = 8
    """the number of parallel game environments"""
    agent: ValidAgents = "llm_rules_agent"
    """the agent to use"""
    parallel_pipeline: bool = True
    """if toggled, the pipeline will be parallelized"""
    llm: ValidModels = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    """the language model to use"""

    # eval
    num_steps: int = 64
    """the number of steps to run in each eval environment per policy rollout"""
    num_iterations: int = 5
    """the number of eval iterations"""

    # Algorithm
    num_rules: int = 3
    """The number of rules for rule-based LLM-only agent"""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
        return None


def parallel_pipeline(lang_agent, state_text, state_vector=None, post_action=True):
    # Get the action and value in parallel
    def call_pipeline(i):
        with torch.no_grad():
            return lang_agent(
                state_text=state_text[i],
                state_vector=state_vector[i] if state_vector is not None else None,
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


def main(args: Args):
    def make_env(env_id, eval=False):
        def symlog(x: float) -> float:
            import math

            return math.copysign(math.log1p(abs(x)), x)

        def thunk():
            if eval:
                env = gym.make(env_id, max_episode_steps=None, T=args.num_steps)
            else:
                env = gym.make(env_id)

            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.TransformReward(env, symlog)
            return env

        return thunk

    eval_env_funs = [make_env(args.env_id, eval=True) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(eval_env_funs)
    chat_model = get_llm_api(args.llm)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

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

    text_logs_path = f"text_logs/eval_llm/{run_name}.jsonl"
    json_logger_mode = "w"
    os.makedirs(os.path.dirname(text_logs_path), exist_ok=True)
    jsonl_logger = jsonlines.open(text_logs_path, mode=json_logger_mode)

    global_step = 0

    obs, infos = envs.reset()
    _, next_state_text = obs

    # TODO: include iteration in the checkpoint
    for iter in range(1, args.num_iterations + 1):
        transitions = defaultdict(list)
        for step in tqdm(
            range(args.num_steps), desc=f"Iter: {iter},  Gathering trajectories"
        ):
            global_step += args.num_envs

            # Get the action and value in parallel
            if args.parallel_pipeline:
                env_actions, outputs, messages = parallel_pipeline(
                    lang_agent, next_state_text, post_action=True
                )
            else:
                env_actions, outputs, messages = lang_agent(
                    next_state_text, post_action=True
                )

            # Append the rules
            if args.agent == "llm_rules_agent":
                transitions["rules"].append(outputs["rules"])
                transitions["sel_reward_scores"].append(outputs["sel_reward_scores"])
                transitions["sel_reward"].append(outputs["sel_reward"])

            # Step the environment
            obs, rewards, _, _, infos = envs.step(env_actions)
            _, next_state_text = obs

            # Store the reward
            transitions["rewards"].append(list(rewards))

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
                if args.agent == "llm_rules_agent":
                    rules_str = "\n".join(outputs["rules"][0])
                    rules_scores = [
                        f"{k}: {v}"
                        for k, v in outputs["sel_reward_scores_raw"][0].items()
                    ]
                    rules_scores_str = "\n".join(rules_scores)
                    example = (
                        f"{outputs['initial_prompt'][0]}\n"
                        f"### Thoughts\n {outputs['thoughts'][0]}\n"
                        f"### Rules\n {rules_str}\n"
                        f"### Selected Rules Explainability\n{rules_scores_str}\n"
                        f"### Environment Action\n {env_actions[0]}\n"
                        f"### Explanation\n {outputs['explanation'][0]}\n"
                        f"### Explanation rules only\n {outputs['explanation_rule_only'][0]}"
                    )
                elif args.agent == "no_thoughts_agent":
                    example = (
                        f"{outputs['initial_prompt'][0]}\n"
                        f"### Environment Action\n {env_actions[0]}\n"
                        f"### Explanation\n {outputs['explanation'][0]}"
                    )
                elif args.agent == "base_agent":
                    example = (
                        f"{outputs['initial_prompt'][0]}\n"
                        f"### Thoughts\n {outputs['thoughts'][0]}\n"
                        f"### Environment Action\n {env_actions[0]}\n"
                        f"### Explanation\n {outputs['explanation'][0]}"
                        f"### Explanation no thoughts\n {outputs['explanation_no_thoughts'][0]}"
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

        env_rewards = torch.FloatTensor(transitions["rewards"]).to(device)
        env_rewards = env_rewards.mean().item()
        writer.add_scalar("charts/env_return", env_rewards, global_step)

        if args.agent == "llm_rules_agent":
            sel_rule_rewards = np.array(transitions["sel_reward"]).mean().item()
            rewards = sel_rule_rewards + env_rewards
            total_reward = sel_rule_rewards + env_rewards

            writer.add_scalar("charts/sel_rule_return", sel_rule_rewards, global_step)
            writer.add_scalar("charts/total_return", total_reward, global_step)

            sel_reward_scores = np.array(transitions["sel_reward_scores"])
            sel_reward_scores = np.mean(sel_reward_scores, axis=(0, 1))
            for i, x in enumerate(sel_reward_scores):
                writer.add_scalar(f"charts/sel_reward_scores/q{i}", x, global_step)

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
