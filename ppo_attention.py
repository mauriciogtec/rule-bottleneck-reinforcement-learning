# this file is an adaption of the clean rl's PPO implementation for rule-based agents
# rule-based language agents use different internal actions than the environment actions
# in addition, one must keep track of the rules selected by the agent.

import itertools
import logging
import os
import random
import time
from collections import defaultdict

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

import envs as E  # registers the gym environments during import
from agents import RulesSelectorActorCritic
from layers import AttentionActorCritic

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
        return torch.load(checkpoint_path, map_location=device)
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
        return None


@hydra.main(config_path="conf", config_name="ppo", version_base=None)
def main(cfg: DictConfig):
    def make_env():
        def thunk():
            env = gym.make(cfg.env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk

    env_funs = [make_env() for i in range(cfg.num_envs)]
    envs = (
        gym.vector.AsyncVectorEnv(env_funs, shared_memory=False)
        if cfg.parallel
        else gym.vector.SyncVectorEnv(env_funs)
    )

    chat_model = ChatTogether(model=cfg.chat_lm)
    embed_model = TogetherEmbeddings(model=cfg.embedder_lm)

    state_dim = envs.single_observation_space[0].shape[0]
    rule_dim = cfg.embed_dim
    hidden_dim = cfg.hidden_dim
    num_rules = cfg.num_rules

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    run_id = f"{__file__.replace('.py', '')}_{cfg.env_id}__{cfg.exp_name}__{cfg.seed}"
    run_name = run_id if cfg.resume else f"{run_id}_{int(time.time())}"
    params = OmegaConf.to_container(cfg, resolve=True)

    if cfg.track:
        import wandb

        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=params,
            name=run_name,
            monitor_gym=True,
            save_code=cfg.wandb_save_code,
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
        max_rule_combinations=cfg.max_rule_combinations,
    )
    optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.learning_rate, eps=1e-5)
    torchsummary.summary(actor_critic)

    ckpt_path = f"checkpoints/{__file__.replace('.py', '')}/best_{run_id}.state"

    global_step = 0
    start_time = time.time()
    best_total_reward = -float("inf")
    best_model = None

    checkpoint = load_checkpoint(ckpt_path, device)
    if checkpoint:
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

    batch_size = int(cfg.num_envs * cfg.num_steps)
    minibatch_size = int(batch_size // cfg.num_minibatches)
    num_iterations = cfg.total_timesteps // batch_size

    obs, infos = envs.reset()
    next_state_vector, next_state_text = obs
    next_state_vector = torch.tensor(next_state_vector, dtype=torch.float32).to(device)
    next_dones = torch.zeros(cfg.num_envs, dtype=torch.bool).to(device)

    for iter in range(1, num_iterations + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (iter - 1.0) / num_iterations
            lrnow = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        transitions = defaultdict(list)
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs

            # Add states and dones
            transitions["state_vector"].append(next_state_vector)
            transitions["state_text"].append(next_state_text)
            transitions["dones"].append(next_dones)

            # Get the action and value
            with torch.no_grad():
                # Use the term env_actions to differentiate from the internal actions
                # which are the selected rules.
                env_actions, outputs, messages = lang_agent(
                    state_text=next_state_text,
                    state_vector=next_state_vector,
                    post_action=True,
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

            # Store the reward
            transitions["rewards"].append(list(rewards))

            if "episode" in infos:
                for i in range(cfg.num_envs):
                    if infos["_episode"][i]:
                        r, l = infos["episode"]["r"][i], infos["episode"]["l"][i]
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)

                        logging.info(
                            f"global_step={global_step}, episodic_return={r:.4f}"
                        )

            if step == 0 or step % cfg.log_examples_interval == 0:
                # log the final selected rule and explanation
                rules_str = "\n".join(outputs["rules"][0])
                example = (
                    f"{outputs['initial_prompt'][0]}\n"
                    f"### Thoughts\n {outputs['thoughts'][0]}\n"
                    f"### Rules\n {rules_str}\n"
                    f"### Selected Rule\n {outputs['sel_rule'][0]}\n"
                    f"### Selected Rule Probability\n {np.exp(outputs['sel_logprob'][0]):.2f}\n"
                    f"### Selected Rule Reward\n {outputs['sel_reward'][0]}\n"
                    f"### Environment Action\n {env_actions[0]}\n"
                    f"### Explanation\n {outputs['explanation'][0]}"
                )

                conversation = "\n".join(
                    [f"\n\n## {x['role']}\n\n{x['content']}" for x in messages[0]]
                )
                writer.add_text("text/examples", example, global_step)
                writer.add_text("text/conversation", conversation, global_step)

        # convert the transitions to tensors
        state_vector = torch.stack(transitions["state_vector"])
        rules_emb, rules_padding_mask = pad_rules(transitions["rules_emb"])
        dones = torch.stack(transitions["dones"])
        values = torch.stack(transitions["value"])
        logprobs = torch.stack(transitions["sel_logprob"])
        sel_idxs = torch.stack(transitions["sel_idx"])
        rewards = torch.FloatTensor(transitions["rewards"]).to(device)
        sel_rule_rewards = torch.stack(transitions["sel_reward"])
        # rewards += sel_rule_rewards

        # save best model
        total_reward = rewards.mean().item()
        if best_total_reward < total_reward:
            best_total_reward = total_reward
            best_model = actor_critic.state_dict()

        if iter % cfg.save_interval == 0 or iter == num_iterations:
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
            # need to call one last time to get the rules embeddings
            _, outputs, _ = lang_agent(
                state_text=next_state_text,
                state_vector=next_state_vector,
                post_action=True,
            )
            next_value = torch.stack(outputs["value"])

            # compute the advantages
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_dones.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
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
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = lang_agent.get_action_and_value_from_embeddings(
                    b_states_vec[mb_inds],
                    b_rules_emb[mb_inds],
                    rules_padding_mask=b_padding_mask[mb_inds],
                    sel_idxs=b_sel_idxs[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor_critic.parameters(), cfg.max_grad_norm
                )
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
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

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
