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
    checkpoint: str
    """the checkpoint to load the model from"""
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

    # Environment
    env_id: str = "Uganda"
    """the id of the environment"""
    num_envs: int = 4
    """the number of parallel game environments"""
    parallel_pipeline: bool = True
    """if toggled, the pipeline will be parallelized"""
    max_episode_steps: Optional[int] = 32
    """the maximum number of steps per episode"""

    dropout: float = 0.0
    """the dropout rate"""

    # Eval
    eval: bool = True
    """if toggled, the agent will be evaluated"""
    num_eval_episodes: int = 20
    """the number of episodes to evaluate the agent"""
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

def load_checkpoint(checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location=device)
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
        return None


def main(args: Args):
    run_id = f"{args.agent}__{args.env_id}__{args.exp_name}__{args.seed}"
    run_name = run_id if args.resume else f"{run_id}__{int(time.time())}"

    ckpt_path = args.checkpoint
    # text_logs_path = f"text_logs/{run_name}.jsonl"
    # json_logger_mode = "w" if not args.resume else "a"
    # os.makedirs(os.path.dirname(text_logs_path), exist_ok=True)
    # jsonl_logger = jsonlines.open(text_logs_path, mode=json_logger_mode)

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
    lang_agent.deterministic = args.eval_deterministic

    starting_step = 0
    start_time = time.time()
    best_total_reward = -float("inf")
    best_model = None
    best_model_epoch = -1

    checkpoint = load_checkpoint(ckpt_path, dev)
    logging.info(
        f"Resuming training from checkpoint at step {checkpoint['global_step']}."
    )
    actor.load_state_dict(checkpoint["actor_state"])
    qf1.load_state_dict(checkpoint["qf1_state"])
    qf2.load_state_dict(checkpoint["qf2_state"])


    start_time = time.time() - checkpoint["elapsed_time"]
    best_total_reward = checkpoint["best_total_reward"]
    best_model = checkpoint["best_model"]
    best_model_epoch = checkpoint["best_model_epoch"]
    # buffer = checkpoint["buffer"]
    logging.info(f"Resumed training from checkpoint at step {starting_step}.")
    # log best model epoch
    logging.info(f"Loading model from best epoch: {best_model_epoch}")

    best_actor, best_qf1, best_qf2 = best_model
    lang_agent.actor = best_actor
    def critic(rules_emb, obs_vec):
        q1 = best_qf1(rules_emb, obs_vec)
        q2 = best_qf2(rules_emb, obs_vec)
        return torch.min(q1, q2)
    lang_agent.critic = critic


    # logging.info(f"Starting buffer size: {buffer.size()}")

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

    global_step = 0
    with tqdm(total=args.num_eval_episodes) as pbar:
        while global_step < args.num_eval_episodes:
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

                        global_step += 1
                        pbar.update(1)

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

                # # log the conversation and example in jsonl
                # jsonl_logger.write(
                #     {
                #         "global_step": global_step,
                #         "example": example,
                #         "conversation": messages[0],
                #     }
                # )

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

        
    envs.close()
    eval_envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.parse(Args)

    args.agent = "eval-rbrl"
    if not args.thoughts:
        args.agent += "-no-thoughts"
    if not args.in_context_learning:
        args.agent += "-no-in-context"
    
    args.agent += "--" + args.llm

    main(args)
