import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from time import time
from typing import Optional, List
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

valid_agent_lst = ['rule_select_agent', 'base_agent', 'no_thoughts_agent', 'llm_rules_no_thoughts', 'llm_rules_agent']
checkpoint_path = "rbrl-no-in-context__Uganda__v7b__gpt-4o-mini-huit__457__0.state"
# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

@dataclass
class Args:
    rbrl_checkpoint: str
    """the path to the rule-based RL checkpoint"""

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

    agent_list: Optional[List[str]] = None  # to be filled
    """the list of agents to use"""
    use_thoughts_with_rules: bool = False
    """if toggled, the thoughts will be used with rules"""

    # agent: PureLanguageAgents = "base_agent"
    # """the agent to use"""
    parallel_pipeline: bool = True
    """if toggled, the pipeline will be parallelized"""

    llm: ValidLLMs = "gpt-4o-mini-huit"
    """the language model to use"""

    use_thoughts_with_rules: bool = False
    """if toggled, the thoughts will be used with rules"""

    # eval
    num_episodes: int = 8
    """the number of eval iterations"""
    max_episode_steps: int = 16
    """the number of steps per eval iteration"""

    # Algorithm
    num_rules: int = 1
    """The number of rules for rule-based LLM-only agent"""

    ## Embedder
    embedder_lm: str = "togethercomputer/m2-bert-80M-8k-retrieval"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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



def main(args: Args):
    eval_env_funs = [
        make_env(args.env_id, args.seed + i, max_episode_steps=args.max_episode_steps)
        for i in range(args.num_envs)
    ]
    envs = gym.vector.SyncVectorEnv(eval_env_funs)
    chat_model = get_llm_api(args.llm)

    set_seed(args.seed)

    t = int(time())
    run_name = f"survey_agents__{args.env_id}__{args.llm}__{args.exp_name}__{args.seed}__{t}"
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
        table = wandb.Table(columns=["step", "state", *args.agent_list])
                                     
    writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in params.items()])),
    )

    global_step = 0

    obs, infos = envs.reset()
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
    state_dim = envs.single_observation_space[0].shape[-1]
    hidden_dim = 16
    num_rules = 5
    dropout_val = 0.05
    proj_type_input = "linear" 

    actor = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout=dropout_val,
        proj_type=proj_type_input,
    )
    qf1 = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout= dropout_val,
        proj_type=proj_type_input,
    )
    qf2 = CrossAttentionNetwork(
        q_dim=rule_dim,
        k_dim=state_dim,
        hidden_dim=hidden_dim,
        dropout= dropout_val,
        proj_type=proj_type_input,
    )
    # Define critic function
    def critic(rules_emb, obs_vec):
        q1 = qf1(rules_emb, obs_vec)
        q2 = qf2(rules_emb, obs_vec)
        return torch.min(q1, q2)
    
    embed_model = TogetherEmbeddings(model=args.embedder_lm)

    while total_episodes < args.num_episodes:
        global_step += 1

        # Collect actions for each agent, randomly pick one to advance the state
        actions_per_agent = {}

        for agent in args.agent_lst:
            if agent == "base_agent":
                lang_agent = agents.BaseAgent(
                task_text=envs.metadata["task_text"],
                action_space_text=envs.metadata["action_space_text"],
                llm=chat_model,
            )
            elif agent == "llm_rules_agent":
                example_rules = envs.envs[0].metadata["example_rules"]
                example_rules = "".join(f"- {x}\n" for x in example_rules)
                lang_agent = agents.LLMRulesAgent(
                    task_text=envs.metadata["task_text"],
                    action_space_text=envs.metadata["action_space_text"],
                    num_rules=args.num_rules,
                    llm=chat_model,
                    example_rules=example_rules,
                )
            elif agent == "llm_rules_no_thoughts":
                example_rules = envs.envs[0].metadata["example_rules"]
                example_rules = "".join(f"- {x}\n" for x in example_rules)
                lang_agent = agents.LLMRulesAgentNoThoughts(
                    task_text=envs.metadata["task_text"],
                    action_space_text=envs.metadata["action_space_text"],
                    num_rules=args.num_rules,
                    llm=chat_model,
                    example_rules=example_rules,
                )
            elif agent == "no_thoughts_agent":
                lang_agent = agents.NoThoughtsAgent(
                    task_text=envs.metadata["task_text"],
                    action_space_text=envs.metadata["action_space_text"],
                    llm=chat_model,
                )
            elif agent == "rule_select_agent":
                example_rules = envs.envs[0].metadata["example_rules"]
                example_rules = "".join(f"- {x}\n" for x in example_rules)
                lang_agent = agents.RulesSelectorActorCritic(
                    actor = actor,
                    task_text=envs.metadata["task_text"],
                    action_space_text=envs.metadata["action_space_text"],
                    num_rules=args.num_rules,
                    example_rules=example_rules,
                    llm=chat_model,
                    embededder=embed_model,
                    critic = critic,
                ) 
                ## TODO: Hi Mauracio, please check do you want other initialization for rule select agent
            else:
                raise ValueError(f"Unknown baseline: {args.agent}")
            
            # Get the action and value in parallel
            if args.parallel_pipeline:
                outputs, messages = lang_agent.parallel_pipeline(next_state_text)
            else:
                outputs, messages = lang_agent(next_state_text)

            # Store actions for this agent
            actions_per_agent[agent] = [x["action"] for x in outputs]

            if agent == "llm_rules_agent" or agent == "rule_select_agent":
                sel_reward_scores = [x["sel_reward_scores"] for x in outputs]
                sel_rewards = [x["sel_reward"] for x in outputs]
            
                # accumulate and log the rewards
                for j in range(args.num_envs):
                    _ep_buffer["sel_rewards_scores"][agent][j].append(sel_reward_scores[j])
                    _ep_buffer["sel_rewards_total"][agent][j].append(sel_rewards[j])

            if step == 0 or step % args.log_examples_interval == 0:
                if agent in ("llm_rules_agent", "llm_rules_no_thoughts"):
                    rules_str = "\n".join(outputs[0]["rules"])
                    rules_scores = [
                        f"{k}: {v}"
                        for k, v in outputs[0]["sel_reward_scores_raw"].items()
                    ]
                    rules_scores_str = "\n".join(rules_scores)
                    thoughts = outputs[0].get("thoughts", None)
                    example = [
                        f"### Agent: {agent}\n"
                        f"{outputs[0]['initial_prompt']}\n",
                        f"### Thoughts\n {thoughts}\n" if thoughts else "",
                        f"### Rules\n {rules_str}\n",
                        f"### Selected Rules Explainability\n{rules_scores_str}\n",
                        f"### Recommended Environment Action by {agent}: {outputs[0]['action']}\n",
                        f"### Explanation\n {outputs[0]['explanation']}\n",
                    ]
                    example = "".join(example)
                elif agent == "no_thoughts_agent":
                    example = (
                        f"### Agent: {agent}\n"
                        f"{outputs[0]['initial_prompt']}\n"
                        f"### Recommended Environment Action by {agent}: {outputs[0]['action']}\n"
                        f"### Explanation\n {outputs[0]['explanation']}"
                    )
                elif agent == "base_agent":
                    example = (
                        f"### Agent: {agent}\n"
                        f"{outputs[0]['initial_prompt']}\n"
                        f"### Thoughts\n {outputs[0]['thoughts']}\n"
                        f"### Recommended Environment Action by {agent}: {outputs[0]['action']}\n"
                        f"### Explanation\n {outputs[0]['explanation']}"
                    )
                elif agent == "rule_select_agent":
                    example = (
                        f"### Agent: {agent}\n"
                        f"{outputs[0]['initial_prompt']}\n"
                        f"### Thoughts\n {outputs[0]['thoughts']}\n"
                        f"### Recommended Environment Action by {agent}: {outputs[0]['action']}\n"
                        f"### Explanation\n {outputs[0]['explanation']}"
                    )
                ## TODO: Hi Mauracio, I will let you decide what else our rule select agent needs to output

                conversation = "\n".join(
                    [f"\n\n## {x['role']}\n\n{x['content']}" for x in messages[0]]
                )
                writer.add_text(f"{agent}/text/examples", example, global_step)
                writer.add_text(f"{agent}/llm_prompts/conversation", conversation, global_step)

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
                        f"Agent: {agent}, global_step={global_step}, total_episodes={total_episodes}/{args.num_episodes}"
                )
        
        # Randomly select one agent's action to step the environment
        selected_agent = random.choice(args.agent_lst)
        env_actions = actions_per_agent[selected_agent]
        obs, env_rewards, terminations, truncations, infos = envs.step(env_actions)
        _, next_state_text = obs
        # Log which agent's actions were chosen
        logging.info(f"Selected Agent to interact with the environment: {selected_agent}, Actions: {env_actions}")

        # Step 3: Process terminations and log rewards
        for j in range(args.num_envs):
            done_now = terminations[j] or truncations[j]
            if done_now:
                if agent == "llm_rules_agent" or agent == "rule_select_agent":
                    m = np.mean(_ep_buffer["sel_rewards_scores"][agent][j], axis=0)
                    for i, x in enumerate(m):
                        writer.add_scalar(
                            f"charts/{agent}_sel_reward_scores/q{i}", x, global_step
                        )
                    writer.add_scalar(f"{agent}/charts/episodic_sel_rewards", m, global_step)
                    _ep_buffer["sel_rewards_scores"][agent][j].clear()
                    _ep_buffer["sel_rewards_total"][agent][j].clear()
                    _ep_buffer["total_rewards"][agent][j].clear()

                total_episodes += 1
                
        step += 1

    envs.close()
    writer.close()
            

if __name__ == "__main__":
    args = tyro.cli(Args)

    # make agent list
    args.agent_lst = ["base_agent"]
    rbrl_agent = "rbrl"
    if args.use_thoughts_with_rules:
        args.agent_lst.append("llm_rules_agent")
    else:
        args.agent_lst.append("llm_rules_no_thoughts")
        rbrl_agent += "-no-thoughts"
    
    args.agent_lst.append(rbrl_agent)

    main(args)

           
                
     







