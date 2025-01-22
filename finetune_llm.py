# this file is an adaption of the clean rl's PPO implementation for rule-based agents
# rule-based language agents use different internal actions than the environment actions
# in addition, one must keep track of the rules selected by the agent.

from dataclasses import dataclass
import itertools
import logging
import re
import shutil
import os
import random
import warnings
import tyro
import time
from collections import defaultdict, deque
from typing import List, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from torch.nested import nested_tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from llm_apis import llama_prompt_from_messages

from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from bitsandbytes.optim import PagedAdamW8bit
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from trl.models import AutoModelForCausalLMWithValueHead

import envs as E  # registers the gym environments during import

# from agents import RulesSelectorActorCritic
from agents import LLMFineTuningAgent
import jsonlines

# configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# class ActorCritic(nn.Module, AutoModelForCausalLM):
#   """Agent class modeled after transformers.AutoModelForCausalLMWithValueHead"""
#   transformers_parent_class = AutoModelForCausalLM

#   def __init__(self, model, tokenizer):
#     nn.Module().__init__()
#     self.model = model
#     self.tokenizer = tokenizer

#     # value function related stuff
#     self.hidden_size = self.model.config.hidden_size

#     # define and initialize critic head
#     self.v_head = nn.Linear(
#       self.hidden_size, 1, dtype=self.model.dtype
#     ).to(self.model.device)

#   def forward(self, input_ids, attention_mask):
#     outputs = self.model(
#       input_ids=input_ids,
#       attention_mask=attention_mask,
#       output_hidden_states=True,
#     )

#     # here's a small modification from PPO for RLHF since we want the
#     # last_token_hidden_state not the last_hidden_state for all tokens
#     last_token_hidden_state = outputs.hidden_states[-1]
#     lm_logits = outputs.logits
#     loss = outputs.loss
#     value = self.v_head(last_token_hidden_state).squeeze(-1)

#     return (lm_logits, loss, value)


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


def to_token_data_loader(
    messages: List[List[dict]],
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    minibatch_size: int,
    collator: DataCollatorForSeq2Seq,
) -> DataLoader:
    # tokenize each message
    prompts = [llama_prompt_from_messages(m) for m in messages]
    tokens = [tokenizer(x) for x in prompts]
    # add tokkenizer lengths
    for j, t in enumerate(tokens):
        t["input_len"] = len(prompts[j])
        t["env_num"] = j
        # t["messages"] = messages[j]
        # t["prompts"] = prompts[j]
    loader = DataLoader(
        Dataset.from_list(tokens),
        batch_size=minibatch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=min(len(prompts), os.cpu_count() - 1),
    )

    return accelerator.prepare(loader)


def generate_with_model(model, loader, tokenizer, accelerator, max_new_tokens=256):
    """
    Abstract function to handle text generation using a model and loader.

    Args:
        model: The text generation model (e.g., HuggingFace model).
        loader: DataLoader providing input batches.
        tokenizer: Tokenizer used to decode tokens.
        accelerator: Accelerator for distributed computation (e.g., HuggingFace Accelerator).
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        dict: A dictionary with keys:
            - "input_tokens": List of input token IDs for each example.
            - "output_tokens": List of output token IDs for each example.
            - "generated_text": List of generated text for each example.
            - "logprobs": List of log probabilities for the new tokens.
            - "values": List of value estimates for the new tokens.
    """
    B = len(loader.dataset)
    keys = ["input_tokens", "output_tokens", "generated_text", "logprobs", "values"]
    result = {k: [None for _ in range(B)] for k in keys}

    _outputs = []
    with torch.no_grad():
        for batch in loader:
            genout = model.generate(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                return_dict_in_generate=True,
                output_logits=True,
                output_hidden_states=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            all_genout = accelerator.gather(genout)
            all_batches = accelerator.gather(batch)
            _outputs.append((all_genout, all_batches))

    # Process generated outputs and map to results
    for genout, b in _outputs:
        logits = torch.stack(genout.logits, dim=1)
        bsize = logits.shape[0]
        num_new_tokens = logits.shape[1]
        input_tokens = b.input_ids
        new_tokens = genout.sequences[:, -num_new_tokens:]
        logprobs = F.log_softmax(logits, dim=-1)
        sel_logprobs = logprobs.gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)

        # Extract last hidden states for v_head evaluation
        # The first -1 is for the last layer, the second -1 is a trick
        # for index 0, it means the last input token, for the remaining, the second dimension
        # is always 1, so we can just index it with -1
        last_hidden_states = [x[-1][:, -1] for x in genout.hidden_states]
        last_hidden_states = torch.stack(last_hidden_states, dim=1)
        value_estimates = model.v_head(last_hidden_states).squeeze(-1)

        for j in range(bsize):
            # Store results in the appropriate index
            env = b.env_num[j]
            new_text = tokenizer.decode(new_tokens[j], skip_special_tokens=True)
            result["input_tokens"][env] = input_tokens[j]
            result["output_tokens"][env] = new_tokens[j]
            result["generated_text"][env] = new_text
            result["logprobs"][env] = sel_logprobs[j]
            result["values"][env] = value_estimates[j]

    return result


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


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
    cuda: bool = True
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
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    # num_minibatches: int = 16
    # """the number of mini-batches"""
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
    num_rules: int = 10
    """The number of rules for rule-based LLM-only agent"""
    """the language model to use"""
    hidden_dim: int = 16
    """the hidden dimension of the networks"""
    parallel_pipeline: bool = True
    """if toggled, the pipeline will be parallelized"""
    llm: str = "meta-llama/Llama-3.2-3B-Instruct"
    """The model to finetune"""
    train_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    """The dtype to use for training"""
    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    minibatch_size: int = 2
    """The minibatch size"""


# def make_env(env_id, seed, eval=False):
#     def thunk():
#         if eval:
#             env = gym.make(env_id, max_episode_steps=None)
#         else:
#             env = gym.make(env_id)

#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         # env = E.wrappers.SymlogRewardsWrapper(env)
#         env.reset(seed=seed)
#         return env

#     return thunk

THOUGHT_PROMPT = (
    "First, reason about what elements should be considered when choosing the optimal action"
    " in the given task of the decision making agent."
    " Your response should consist of a single paragraph that reflects on the consequences, benefits, and drawbacks"
    " of each action in the current state. Conclude the paragraph with a reflection of how they inform the design"
    " of the priorization rules, and the different types of priorization rules that could be applied to the given scenario."
)

ACTION_PROMPT = (
    "Now, choose the optimal action given the current problem state. "
    "Do not provide additional information or context for your answer, only the action. "
    f"\n\n### Possible actions:\n\n"
)


def main(args: Args):
    run_id = f"ppo_attention_{args.env_id}__{args.exp_name}__{args.llm}__{args.seed}"
    run_name = run_id if args.resume else f"{run_id}__{int(time.time())}"

    ckpt_path = f"checkpoints/best_{run_name}.state"
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

    wrappers = [gym.wrappers.RecordEpisodeStatistics]
    envs = gym.make_vec(args.env_id, args.num_envs, wrappers=wrappers)

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_id = f"ppo_attention_{args.env_id}__{args.exp_name}__{args.seed}"
    run_name = run_id if args.resume else f"{run_id}_{int(time.time())}"

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

    # Load HF model
    dtype = getattr(torch, args.train_dtype)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=1,
        lora_alpha=2,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.llm,
        quantization_config=quantization_config,
        peft_config=peft_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_cache=True,
    )

    print_trainable_parameters(model)

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    collator = DataCollatorForSeq2Seq(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    lang_agent = LLMFineTuningAgent(
        task_text=envs.metadata["task_text"],
        action_space_text=envs.metadata["action_space_text"],
        llm=model,
        tokenizer=tokenizer,
    )

    # Optimizer (8 bit variant)
    optimizer = PagedAdamW8bit(
        model.parameters(),
        lr=args.learning_rate,
    )

    # Accelerator to handle parallelism
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    model, optimizer = accelerator.prepare(model, optimizer)

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
    minibatch_size = args.minibatch_size
    num_iterations = args.total_timesteps // batch_size

    obs, infos = envs.reset()
    _, state_text = obs

    iter_start = (global_step // args.total_timesteps) + 1

    # keep logging buffers for the rewards
    dones = [False for _ in range(args.num_envs)]
    autoreset = np.zeros(args.num_envs, dtype=bool)
    _ep_buffer = defaultdict(lambda: [[] for _ in range(args.num_envs)])
    _rolling_returns = deque(maxlen=args.rolling_returns_window)

    for iter in range(iter_start, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iter - 1.0) / num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        b_value = [[] for _ in range(args.num_envs)]
        b_logprob = [[] for _ in range(args.num_envs)]
        b_input_tokens = [[] for _ in range(args.num_envs)]
        b_output_tokens = [[] for _ in range(args.num_envs)]
        b_done = [[] for _ in range(args.num_envs)]
        b_reward = [[] for _ in range(args.num_envs)]
        b_advantage = [[] for _ in range(args.num_envs)]
        b_returns = [[] for _ in range(args.num_envs)]

        for step in tqdm(
            range(args.num_steps), desc=f"Iter: {iter},  Gathering trajectories"
        ):
            global_step += args.num_envs

            # === Rollout ===

            # During rollout, we have to collect the following quantities:
            # logouts and values must be the aligned with the input/output tokens
            # thoughts_str = [None for _ in range(args.num_envs)]
            # action_str = [None for _ in range(args.num_envs)]
            # env_actions = [None for _ in range(args.num_envs)]

            # values = [[] for _ in range(args.num_envs)]
            # all_rewards = [[] for _ in range(args.num_envs)]
            # all_dones = [[] for _ in range(args.num_envs)]
            # input_tokens = [[] for _ in range(args.num_envs)]

            # 1. Generate thoughts

            # Get the initial prompts
            init_prompts = [lang_agent.system_prompt_with_state(s) for s in state_text]
            outputs = [
                {"initial_prompt": p, "state_text": s}
                for (p, s) in zip(init_prompts, state_text)
            ]
            messages = [[{"role": "system", "content": p}] for p in outputs]

            # Add the rule generation prompt
            for m in messages:
                m.append({"role": "user", "content": THOUGHT_PROMPT})

            # Tokenize each input state and concert torch dataset
            loader = to_token_data_loader(
                messages, tokenizer, accelerator, minibatch_size, collator
            )
            with torch.no_grad():
                results = generate_with_model(
                    model, loader, tokenizer, accelerator, max_new_tokens=256
                )

            thoughts = []
            # for j in range(args.num_envs):
            #     # get sizes of generation
            #     input_len = len(results["input_tokens"][j])
            #     out_len = len(results["output_tokens"][j])
            #     total_len = input_len + out_len

            #     # create a mask for the generation
            #     gen_mask = torch.zeros(total_len, dtype=torch.bool)
            #     gen_mask[input_len:] = True

            #     # assign reward 0 everywhere (we will assign a reward in the very last step)
            #     b_reward[j].append(torch.zeros(out_len, dtype=dtype))

            #     # assign the logprobs / add zeros for the input tokens
            #     logprobs = torch.zeros(total_len, dtype=dtype)
            #     logprobs[gen_mask] = results["logprobs"][j]

            #     # assign the values  with zeros for the input tokens
            #     values = torch.zeros(total_len, dtype=dtype)
            #     values[gen_mask] = results["values"][j]

            for j in range(args.num_envs):
                if not autoreset[j]:
                    b_value[j].append(results["values"][j])
                    b_input_tokens[j].append(results["input_tokens"][j])
                    b_output_tokens[j].append(results["output_tokens"][j])
                    b_logprob[j].append(results["logprobs"][j])
                    b_reward[j].append(torch.zeros_like(results["values"][j]))
                    training_done = torch.zeros_like(results["values"][j])
                    training_done[0] = dones[j]
                    b_done[j].append(training_done)
                    messages[j].append(
                        {"role": "assistant", "content": results["generated_text"][j]}
                    )
                thoughts.append(results["generated_text"][j])

            for i, gen_text in enumerate(results["generated_text"]):
                messages[i].append({"role": "assistant", "content": gen_text})

            # 2. Get the action
            act_space_text = envs.metadata["action_space_text"]
            for m in messages:
                m.append({"role": "user", "content": ACTION_PROMPT + act_space_text})

            loader = to_token_data_loader(
                messages, tokenizer, accelerator, minibatch_size, collator
            )
            with torch.no_grad():
                results = generate_with_model(
                    model, loader, tokenizer, accelerator, max_new_tokens=256
                )

            env_actions = [re.findall(r"\d+", x)[0] for x in results["generated_text"]]

            # Step the environment
            next_obs, env_rewards, dones, truncations, infos = envs.step(env_actions)
            _, next_state_text = next_obs

            for j in range(args.num_envs):
                if not autoreset[j]:
                    b_value[j].append(results["values"][j])
                    b_input_tokens[j].append(results["input_tokens"][j])
                    b_output_tokens[j].append(results["output_tokens"][j])
                    b_logprob[j].append(results["logprobs"][j])
                    training_rewards = torch.zeros_like(results["values"][j])
                    training_rewards[-1] = env_rewards[j]
                    b_reward[j].append(training_rewards)
                    b_done[j].append(torch.zeros_like(results["values"][j]))
                    messages[j].append(
                        {"role": "assistant", "content": results["generated_text"][j]}
                    )

            # add the transition to the buffer
            for j in range(args.num_envs):
                # if not autoreset[j]:
                #     b_value[j].append(values[j])
                #     b_logprob[j].append(logprobs[j])
                #     b_input_tokens[j].append(input_tokens[j])
                #     b_output_tokens[j].append(output_tokens[j])
                #     b_done[j].append(torch.tensor(dones[j], dtype=dtype))
                #     b_reward[j].append(torch.tensor(env_rewards[j], dtype=dtype))

                done_now = dones[j] or truncations[j]
                if not done_now:
                    _ep_buffer["env_rewards"][j].append(env_rewards[j].item())
                    prob = results["logprobs"][j].exp().mean().item()
                    _ep_buffer["sel_probs"][j].append(prob)
                else:
                    # log the rewards
                    writer.add_scalar(
                        f"charts/episodic_env_rewards",
                        np.mean(_ep_buffer["env_rewards"][j]).item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_sel_probs",
                        np.mean(_ep_buffer["sel_probs"][j]).item(),
                        global_step,
                    )

                    # flush
                    _ep_buffer["env_rewards"][j].clear()
                    _ep_buffer["sel_probs"][j].clear()

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
                example = (
                    f"{outputs[0]['initial_prompt']}\n"
                    f"### Thoughts\n{thoughts[0]}\n"
                    f"### Environment Action\n{env_actions[0]}\n"
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

                # log the conversation and example in jsonl
                jsonl_logger.write(
                    {
                        "global_step": global_step,
                        "example": example,
                        "conversation": conversation,
                    }
                )

            # advance
            autoreset = np.logical_or(dones, truncations)
            obs = next_obs
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

        # Get next value
        init_prompts = [lang_agent.system_prompt_with_state(s) for s in state_text]
        messages = [[{"role": "system", "content": p}] for p in outputs]
        loader = to_token_data_loader(
            messages, tokenizer, accelerator, minibatch_size, collator
        )
        next_values = []
        with torch.no_grad():
            for batch in loader:
                _, _, _values = model(
                    batch.input_ids, attention_mask=batch.attention_mask
                )
                all_values = accelerator.gather(_values)
                next_values.extend(all_values)

        next_values = torch.stack(next_values)[:, -1]  # keep only the last value token
        next_dones = torch.tensor(dones, dtype=dtype).to(dev)

        # We now need to perform a batch expansion step .
        # It is as follows
        # 1. For every input, output tokens pair, Concatenate the pair input + output, creating a mask indicating which one is which
        # 2. It's corresponding reward is assigned t
        # o the last token of this sequence, with zeros in all other cases

        # bootstrap value if not done, only
        with torch.no_grad():
            # compute the advantages
            # need to do per environment to deal with potentially different lengths
            # as well as nested tensors for the rules
            # (and maybe future support for different state space shapes)

            _collated_value = [None for _ in range(args.num_envs)]
            _collated_reward = [None for _ in range(args.num_envs)]
            _collated_done = [None for _ in range(args.num_envs)]
            _collated_return = [None for _ in range(args.num_envs)]
            _collated_advantage = [None for _ in range(args.num_envs)]

            for j in range(args.num_envs):
                _collated_value[j] = torch.cat(b_value[j])
                _collated_reward[j] = torch.cat(b_reward[j])
                _collated_done[j] = torch.cat(b_done[j])
                _collated_return[j] = torch.zeros_like(_collated_value[j])
                _collated_advantage[j] = torch.zeros_like(_collated_value[j])
                T = _collated_value[j].shape[0]

                # it is not equal to num steps due to the autoreset skip
                lastgaelam = 0
                for t in reversed(range(T)):
                    if t == T - 1:
                        nextnonterminal = 1.0 - next_dones[j]
                        nextvalues = next_values[j]
                    else:
                        nextnonterminal = 1.0 - _collated_done[j][t + 1]
                        nextvalues = _collated_value[j][t + 1]
                    delta = (
                        _collated_reward[j][t]
                        + args.gamma * nextvalues * nextnonterminal
                        - _collated_value[j][t]
                    )
                    _collated_advantage[j][t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                    _collated_return[j][t] = (
                        _collated_advantage[j][t] + _collated_value[j][t]
                    )

                # extract b_advantage and b_return by splitting by sizes
                sizes = [len(x) for x in b_value[j]]
                b_advantage[j] = torch.split(_collated_advantage[j], sizes)
                b_returns[j] = torch.split(_collated_return[j], sizes)

        # Now collate across environments
        b_value = list(itertools.chain(*b_value))
        b_logprob = list(itertools.chain(*b_logprob))
        b_input_tokens = list(itertools.chain(*b_input_tokens))
        b_output_tokens = list(itertools.chain(*b_output_tokens))
        b_done = list(itertools.chain(*b_done))
        b_reward = list(itertools.chain(*b_reward))
        b_advantage = list(itertools.chain(*b_advantage))
        b_returns = list(itertools.chain(*b_returns))

        def _fill_zeros(x, idx):
            out = torch.zeros_like(idx, dtype=x.dtype).to(x.device)
            out[idx.bool()] = x
            return out

        # Dataset from token sequences
        dset = []
        for j in range(len(b_value)):
            input_ids = torch.cat([b_input_tokens[j], b_output_tokens[j]])[:-1]
            labels = torch.cat([b_input_tokens[j], b_output_tokens[j]])[1:]
            prompt_mask = torch.ones_like(input_ids)
            # the -1 below is to collapse all the prompt tokens to one state
            prompt_mask[: (len(b_input_tokens[j]) - 1)] = 0 
            logprob = _fill_zeros(b_logprob[j], prompt_mask)
            done = _fill_zeros(b_done[j], prompt_mask)
            reward = _fill_zeros(b_reward[j], prompt_mask)
            advantage = _fill_zeros(b_advantage[j], prompt_mask)
            returns = _fill_zeros(b_returns[j], prompt_mask)
            value = _fill_zeros(b_value[j], prompt_mask)

            dset.append(
                {
                    "input_ids": input_ids,
                    "output_ids": labels,
                    "attention_mask": prompt_mask,
                    "value": value,
                    "logprob": logprob,
                    "done": done,
                    "reward": reward,
                    "advantage": advantage,
                    "returns": returns,
                }
            )
        dset = Dataset.from_list(dset)
        loader = DataLoader(
            dset,
            batch_size=minibatch_size,
            shuffle=True,
            collate_fn=collator,
        )
        loader = accelerator.prepare(loader)

        # Optimizing the policy and value network
        # b_inds = np.arange(N, dtype=np.int32)
        clipfracs = []
        for epoch in tqdm(range(args.update_epochs), desc=f"Iter: {iter},  Optimizing"):
            # np.random.shuffle(b_inds)
            # for start in range(0, N, minibatch_size):
            for batch in loader:
                # end = start + minibatch_size
                # mb_inds = b_inds[start:end]

                # mb_input_tokens = [b_input_tokens[i] for i in mb_inds]
                # mb_output_tokens = [b_output_tokens[i] for i in mb_inds]
                # mb_done = [b_done[i] for i in mb_inds]
                # mb_reward = [b_reward[i] for i in mb_inds]
                # mb_advantage = [b_advantage[i] for i in mb_inds]

                # get the new policy
                # cat input/output tokens and evaluate
                new_logits, loss, new_value = model(
                    batch.input_ids,
                    attention_mask=batch.attention_mask,
                )

                gather_ixs = batch.output_ids.unsqueeze(-1)
                dist = torch.distributions.Categorical(logits=new_logits)
                new_entropy = dist.entropy()
                new_logprob = F.log_softmax(new_logits, dim=-1)
                new_logprob = new_logprob.gather(-1, gather_ixs).squeeze(-1)

                logratio = new_logprob - batch.logprob
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = batch.advantage
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2)
                pg_loss = torch.mean(batch.input_mask * pg_loss)

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = (new_value - batch.returns) ** 2
                    v_clipped = batch.value + torch.clamp(
                        new_value - batch.value,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - batch.value) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * torch.mean(batch.input_mask * v_loss_max)
                else:
                    v_loss = 0.5 * torch.mean(
                        batch.input_mask * (new_value - batch.returns) ** 2
                    )

                entropy_loss = torch.mean(batch.input_mask * new_entropy)
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_value.cpu().numpy(), b_returns.cpu().numpy()
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
