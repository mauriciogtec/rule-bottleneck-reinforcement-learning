import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from time import time
import pandas as pd
from typing import Optional, List

from tqdm import tqdm
from layers import CrossAttentionNetwork

import gymnasium as gym
import numpy as np
import torch
import torch.optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from langchain_together import TogetherEmbeddings

import agents
from agents import PureLanguageAgents
import envs as E  # registers the gym environments during import
from llm_apis import get_llm_api, ValidLLMs

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import jsonlines

valid_agent_lst = [
    "rule_select_agent",
    "base_agent",
    "no_thoughts_agent",
    "llm_rules_no_thoughts",
    "llm_rules_agent",
]


def load_checkpoint(checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
        return None


@dataclass
class Args:
    rbrl_checkpoint: str
    """the path to the rule-bottleneck RL checkpoint"""
    tbrl_checkpoint: str
    """the path to the thought-bottleneck RL checkpoint"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rulebots-compare"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    log_examples_interval: int = 10
    """the interval to log examples"""
    wandb_save_code: bool = True
    """if toggled, the code will be saved to wandb"""
    dropout: float = 0.0
    """the dropout rate"""

    # Environment
    env_id: str = "Uganda"
    """the id of the environment"""
    num_envs: int = 4
    """the number of parallel game environments"""

    agent_list: Optional[List[str]] = None  # to be filled
    """the list of agents to use"""

    llm: ValidLLMs = "gpt-4o-mini-huit"
    """the language model to use"""

    # use_thoughts_with_rules: bool = False
    # """if toggled, the thoughts will be used with rules"""

    # eval
    num_episodes: int = 10
    """the number of eval iterations"""
    max_episode_steps: int = 16
    """the number of steps per eval iteration"""

    # Algorithm
    num_rules: int = 5
    """The number of rules for rule-based LLM-only agent"""

    ## Embedder
    embedder_lm: str = "togethercomputer/m2-bert-80M-8k-retrieval"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_env(env_id, seed, max_episode_steps=None):
    # def scale_reward(r):
    #     return r / max_episode_steps

    def thunk():
        env = gym.make(env_id)
        # if env_id == "HeatAlerts":
        #     env = gym.wrappers.TransformReward(env, func=scale_reward)
        if env_id not in ["BinPacking", "BinPackingIncremental"]:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env

    return thunk


def main(args: Args):
    env = make_env(args.env_id, args.seed, max_episode_steps=args.max_episode_steps)()
    chat_model = get_llm_api(args.llm)

    set_seed(args.seed)

    t = int(time())
    run_name = (
        f"survey_agents__{args.env_id}__{args.llm}__{args.exp_name}__{args.seed}__{t}"
    )
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

    global_step = 0

    obs, infos = env.reset()
    _, next_state_text = obs

    ## We need to create different buffers for different agents
    _ep_buffer = defaultdict(
        lambda: {agent: [[] for _ in range(args.num_envs)] for agent in args.agent_lst}
    )

    total_episodes = 0
    step = 0
    autoreset = np.zeros(args.num_envs, dtype=bool)

    ## Initializatio  needed for Rule Select Agent
    rule_dim = embed_dim = 768
    state_dim = env.observation_space[0].shape[-1]
    hidden_dim = 16
    num_rules = 5

    proj_type_input = "linear"

    actor = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        proj_type=proj_type_input,
    )
    qf1 = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        proj_type=proj_type_input,
    )
    qf2 = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        proj_type=proj_type_input,
    )
    actor.eval()
    qf1.eval()
    qf2.eval()

    # Define critic function
    def critic(rules_emb, obs_vec):
        q1 = qf1(rules_emb, obs_vec)
        q2 = qf2(rules_emb, obs_vec)
        return torch.min(q1, q2)

    embed_model = TogetherEmbeddings(model=args.embedder_lm)

    lang_agents = {}
    for agent in args.agent_list:
        if agent == "base_agent":
            lang_agent = agents.BaseAgent(
                task_text=env.metadata["task_text"],
                action_space_text=env.metadata["action_space_text"],
                llm=chat_model,
            )
        elif agent == "llm_rules_agent":
            example_rules = env.metadata["example_rules"]
            example_rules = "".join(f"- {x}\n" for x in example_rules)
            lang_agent = agents.LLMRulesAgent(
                task_text=env.metadata["task_text"],
                action_space_text=env.metadata["action_space_text"],
                num_rules=1,
                llm=chat_model,
                example_rules=example_rules,
            )
        elif agent == "llm_rules_no_thoughts":
            example_rules = env.metadata["example_rules"]
            example_rules = "".join(f"- {x}\n" for x in example_rules)
            lang_agent = agents.LLMRulesAgentNoThoughts(
                task_text=env.metadata["task_text"],
                action_space_text=env.metadata["action_space_text"],
                num_rules=1,
                llm=chat_model,
                example_rules=example_rules,
            )
        elif agent == "no_thoughts_agent":
            lang_agent = agents.NoThoughtsAgent(
                task_text=env.metadata["task_text"],
                action_space_text=env.metadata["action_space_text"],
                llm=chat_model,
            )
        elif agent.startswith("rbrl") or agent.startswith("tbrl"):
            checkpoint = load_checkpoint(args.rbrl_checkpoint, "cpu")
            # logging.info(
            #     f"Resuming training from checkpoint at step {checkpoint['global_step']}."
            # )
            actor.load_state_dict(checkpoint["actor_state"])
            qf1.load_state_dict(checkpoint["qf1_state"])
            qf2.load_state_dict(checkpoint["qf2_state"])

            best_total_reward = checkpoint["best_total_reward"]
            best_model = checkpoint["best_model"]
            best_model_epoch = checkpoint["best_model_epoch"]
            # buffer = checkpoint["buffer"]
            # log best model epoch
            logging.info(f"Loading model from best epoch: {best_model_epoch}")

            example_rules = env.metadata["example_rules"]
            example_rules = "".join(f"- {x}\n" for x in example_rules)
            lang_agent = agents.RulesSelectorActorCritic(
                actor=actor,
                task_text=env.metadata["task_text"],
                action_space_text=env.metadata["action_space_text"],
                num_rules=args.num_rules,
                example_rules=example_rules,
                llm=chat_model,
                embededder=embed_model,
                use_thoughts="no-thoughts" not in agent,
                optimize_thoughts_only=agent.startswith("tbrl"),
                critic=critic,
            )
            lang_agent.deterministic = True

            best_actor, best_qf1, best_qf2 = best_model
            lang_agent.actor = best_actor

            def critic(rules_emb, obs_vec):
                q1 = best_qf1(rules_emb, obs_vec)
                q2 = best_qf2(rules_emb, obs_vec)
                return torch.min(q1, q2)

            lang_agent.critic = critic

        else:
            raise ValueError(f"Unknown baseline: {args.agent}")
        lang_agents[agent] = lang_agent

    next_state_vec, next_state_text = obs
    next_state_vec = torch.FloatTensor(next_state_vec)

    obs, info = env.reset()
    records = []
    table_columns = [
        "episode",
        "step",
        "state",
        "agent",
        "recommended_action",
        "explanation",
        "thoughts",
        "rules",
        "rule_prob",
        "rule_expl",
        "summary",
        "full_prompt",
        "state_numeric",
        "rules_numeric",
        "sel_rule_idx",
    ]
    with tqdm(total=args.num_episodes) as pbar:
        while total_episodes < args.num_episodes:
            global_step += 1

            # Collect actions for each agent, randomly pick one to advance the state
            actions_per_agent = {}
            examples_per_agent = {}

            for agent, lang_agent in lang_agents.items():
                # Get the action and value in parallel
                outputs, messages = lang_agent.pipeline(
                    state_text=next_state_text, state_vector=next_state_vec
                )

                thoughts = outputs.get("thoughts", "N/A")
                if "rules" in outputs:
                    # rules_str = "\n".join(outputs["rules"])
                    rules_scores = [
                        f"{k}: {v}" for k, v in outputs["sel_reward_scores_raw"].items()
                    ]
                    rules_scores_str = "\n".join(rules_scores)
                    if "sel_logprob" in outputs:
                        rule_prob_str = f"{outputs['sel_logprob'].exp():.2f}"
                    else:
                        rule_prob_str = "N/A"
                else:
                    # rules_str = "N/A"
                    rules_scores_str = "N/A"
                    rule_prob_str = "N/A"

                rules_str = outputs.get("rules", [])
                rules_emb = outputs.get("rules_emb", [])
                sel_rule_idx = outputs.get("sel_idx", -1)

                # Store actions for this agent
                parsed_action = env.metadata["action_parser"](outputs["action"])
                actions_per_agent[agent] = parsed_action

                if agent in ("llm_rules_agent", "llm_rules_no_thoughts"):
                    example = (
                        f"### Action\n {outputs['action']}\n"
                        f"### Thoughts\n {thoughts}\n"
                        f"### Rules\n {rules_str}\n"
                        f"### Selected Rules Explainability\n{rules_scores_str}\n"
                        f"### Explanation\n {outputs['explanation']}\n"
                    )
                    example = "".join(example)
                elif agent == "no_thoughts_agent":
                    example = (
                        f"### Action\n {outputs['action']}\n"
                        f"### Explanation\n {outputs['explanation']}"
                    )
                elif agent == "base_agent":
                    example = (
                        f"### Thoughts\n {outputs['thoughts']}\n"
                        f"### Action\n {outputs['action']}\n"
                        f"### Explanation\n {outputs['explanation']}"
                    )
                elif agent in ("rbrl", "rbrl-no-thoughts"):
                    example = (
                        f"### Thoughts\n {thoughts}\n"
                        f"### Action\n {outputs['action']}\n"
                        f"### Rules\n {rules_str}\n"
                        f"### Selected Rule\n{outputs['sel_rule']}\n"
                        f"### Selected Rule Probability\n{outputs['sel_logprob'].exp():.2f}\n"
                        f"### Selected Rule Reward\n {outputs['sel_reward']}\n"
                        f"### Selected Rule Explainability\n{rules_scores_str}\n"
                        f"### Explanation\n {outputs['explanation']}"
                    )

                examples_per_agent[agent] = example

                if args.track:
                    records.append(
                        (
                            str(total_episodes),
                            str(global_step),
                            next_state_text,
                            agent,
                            str(parsed_action),
                            outputs["explanation"],
                            thoughts,
                            json.dumps(rules_str),
                            rule_prob_str,
                            rules_scores_str,
                            example,
                            outputs["initial_prompt"],
                            json.dumps([float(x) for x in next_state_vec.numpy()]),
                            json.dumps([x.numpy().tolist() for x in rules_emb]),
                            int(sel_rule_idx),
                        )
                    )

            if args.track:
                df = pd.DataFrame(records, columns=table_columns)
                wandb.log({"examples": wandb.Table(dataframe=df)})

            # Randomly select one agent's action to step the environment
            selected_agent = random.choice(args.agent_list)
            env_action = actions_per_agent[selected_agent]
            obs, env_reward, termination, truncation, infos = env.step(env_action)

            if termination or truncation:
                obs, info = env.reset()
                total_episodes += 1
                pbar.update(1)

            next_state_vec, next_state_text = obs
            next_state_vec = torch.FloatTensor(next_state_vec)

            # Log which agent's actions were chosen
            logging.info(
                f"Selected Agent to interact with the environment: {selected_agent}, Actions: {env_action}"
            )

    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)

    # make agent list
    # args.agent_list = ["base_agent", "no_thoughts_agent", "llm_rules_agent",  "rbrl", "rbrl-no-thoughts"]
    args.agent_list = ["base_agent", "tbrl", "rbrl"]

    main(args)
